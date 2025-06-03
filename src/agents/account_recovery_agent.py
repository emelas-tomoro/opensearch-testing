from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
import json

from opensearchpy import OpenSearch

from src.utils.search_utils import GameAccountSearcher, QueryType

client = OpenAI()

class AccountParams(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the account")
    alliance_name: Optional[str] = Field(default=None, description="The name of the alliance")
    updated: Optional[str] = Field(default=None, description="The last updated date of the account")
    create_country: Optional[str] = Field(default=None, description="The country of the account")
    exp_level: Optional[str] = Field(default=None, description="The experience level of the account")

class AccountRecoveryAgent:
    def __init__(self, searcher, min_threshold: int = 100, max_iterations: int = 3):
        self.searcher = searcher
        self.min_threshold = min_threshold
        self.max_iterations = max_iterations
        self.field_rankings = {
            'name': 1,
            'alliance_name': 2,
            'updated': 3,
            'create_country': 4,
            'exp_level': 5
        }
        self.conversation_history = []
        self.search_history = []
        self.current_params = AccountParams()
        
    def get_system_prompt(self, iteration: int, last_hits: int = None, trend: str = None) -> str:
        trend_info = ""
        if last_hits is not None:
            trend_info = f"\nLast search returned {last_hits} results."
            if trend:
                trend_info += f" Results are {trend}."
        
        return f"""
You are a game account search assistant helping a user find their missing account.
Your goal is to gather information systematically based on field importance rankings.

Field Rankings (1=most important, 5=least important):
{json.dumps(self.field_rankings, indent=2)}

Current iteration: {iteration + 1}/{self.max_iterations}
{trend_info}

Instructions:
1. Extract any account information from the user's message.
2. If one of the fields seem vague or generic, don't be afraid to prompt again for the same information.
3. If results are decreasing significantly, ask more specific clarifying questions
4. Focus on the highest-ranked fields first (name is most important)
5. Ask one focused question at a time to refine the search
6. Be conversational and helpful

Current known information:
{self._format_current_params()}
"""

    def _format_current_params(self) -> str:
        params_dict = self.current_params.model_dump(exclude_none=True)
        if not params_dict:
            return "None collected yet"
        return json.dumps(params_dict, indent=2)
    
    def extract_params_from_response(self, user_input: str, iteration: int, last_hits: int = None, trend: str = None) -> Tuple[AccountParams, str]:
        """Extract parameters and get agent's next question"""
        system_prompt = self.get_system_prompt(iteration, last_hits, trend)
        
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # Extract structured data using correct OpenAI API
        param_response = client.responses.parse(
            model="gpt-4.1-mini",
            input=messages,
            text_format=AccountParams,
        )
        
        extracted_params = param_response.output_parsed
        
        # Get conversational response using correct API

        assistant_prompt = f"I found this information: {extracted_params.model_dump(exclude_none=True)}"
        user_prompt = "What should you ask next to help narrow down the search?"

        chat_response = client.responses.create(
            model="gpt-4.1-mini",
            input=messages + [
                {"role": "assistant", "content": assistant_prompt},
                {"role": "user", "content": user_prompt}
            ],

        )
        
        next_question = chat_response.output_text
        return extracted_params, next_question

    def merge_params(self, new_params: AccountParams) -> None:
        """Merge new parameters with existing ones"""
        current_dict = self.current_params.model_dump()
        new_dict = new_params.model_dump(exclude_none=True)
        
        for key, value in new_dict.items():
            if value is not None:
                current_dict[key] = value
        
        self.current_params = AccountParams(**current_dict)

    def build_search_query(self) -> 'GameAccountSearcher':
        """Build search query based on current parameters - creates fresh searcher instance"""
        # Create a new searcher instance for each search
        query_builder = GameAccountSearcher(self.searcher.client)
        
        params_dict = self.current_params.model_dump(exclude_none=True)
        
        # Sort by field rankings (lower number = higher priority)
        sorted_params = sorted(
            params_dict.items(), 
            key=lambda x: self.field_rankings.get(x[0], 999)
        )
        
        for field, value in sorted_params:
            if field == 'name':
                query_builder = query_builder.add_query(field, value, QueryType.MATCH)
            elif field == 'alliance_name':
                query_builder = query_builder.add_query(field, value, QueryType.MATCH, fuzziness="AUTO")
            else:
                query_builder = query_builder.add_query(field, value, QueryType.MATCH)
        
        return query_builder

    def calculate_trend(self, current_hits: int) -> str:
        """Calculate if results are improving or declining"""
        if len(self.search_history) < 1:
            return "initial"
        
        previous_hits = self.search_history[-1]['hits']
        
        if current_hits < previous_hits * 0.5:  # 50% decrease
            return "decreasing significantly"
        elif current_hits < previous_hits:
            return "decreasing slightly"
        elif current_hits > previous_hits * 1.5:  # 50% increase
            return "improving significantly"
        elif current_hits > previous_hits:
            return "improving slightly"
        else:
            return "stable"

    def run_recovery_session(self, initial_query: str) -> Dict[str, Any]:
        """Main recovery session loop"""
        print("🔍 Starting account recovery session...")
        print(f"Minimum threshold: {self.min_threshold} results")
        print(f"Maximum iterations: {self.max_iterations}")
        print("-" * 50)
        
        current_input = initial_query
        
        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}")
            
            # Calculate trend from previous searches
            last_hits = self.search_history[-1]['hits'] if self.search_history else None
            trend = self.calculate_trend(last_hits) if last_hits is not None else None
            
            # Extract parameters and get next question
            new_params, next_question = self.extract_params_from_response(
                current_input, iteration, last_hits, trend
            )
            
            # Merge with existing parameters
            self.merge_params(new_params)
            
            print(f"📝 Current parameters: {self.current_params.model_dump(exclude_none=True)}")
            
            # Perform search
            if any(self.current_params.model_dump(exclude_none=True).values()):
                # try:
                query_builder = self.build_search_query()
                results = query_builder.search()
                print(results)
                
                hits = results['no_of_hits']
                current_trend = self.calculate_trend(hits)
                
                self.search_history.append({
                    'iteration': iteration + 1,
                    'params': self.current_params.model_dump(exclude_none=True),
                    'hits': hits,
                    'trend': current_trend
                })
                
                print(f"🎯 Search results: {hits} hits ({current_trend})")
                
                # Check if we should stop

                if hits == 1:
                    print('Account found!')
                    return {
                        'status': 'success',
                        'final_results': results,
                        'iterations': iteration + 1,
                        'final_params': self.current_params.model_dump(exclude_none=True),
                        'search_history': self.search_history
                    }
                elif hits <= self.min_threshold < 0:
                    print(f"✅ Found {hits} results (below threshold of {self.min_threshold})")
                    return {
                        'status': 'success',
                        'final_results': results,
                        'iterations': iteration + 1,
                        'final_params': self.current_params.model_dump(exclude_none=True),
                        'search_history': self.search_history
                    }
                
                # If results are decreasing significantly, modify the question
                if current_trend == "decreasing significantly":
                    next_question = f"⚠️ Results dropped to {hits} (from {self.search_history[-2]['hits'] if len(self.search_history) > 1 else 'unknown'}). {next_question}"
                
                # except Exception as e:
                #     print(f"❌ Search error: {e}")
                #     next_question = f"There was a search error. {next_question}"
            
            # Add to conversation history
            self.conversation_history.extend([
                {"role": "user", "content": current_input},
                {"role": "assistant", "content": next_question}
            ])
            
            print(f"🤖 Agent: {next_question}")
            
            # Get user input for next iteration
            if iteration < self.max_iterations - 1:  # Don't ask on last iteration
                current_input = input(f"\n👤 Your response: ")
                if not current_input.strip():
                    print("Empty input, stopping session.")
                    break
        
        # Max iterations reached
        final_hits = self.search_history[-1]['hits'] if self.search_history else 0
        return {
            'status': 'max_iterations_reached',
            'final_results': None,
            'iterations': self.max_iterations,
            'final_params': self.current_params.model_dump(exclude_none=True),
            'search_history': self.search_history,
            'final_hits': final_hits
        }

# Example usage with proper GameAccountSearcher integration
def main():
    """
    Example of how to use the AccountRecoveryAgent with GameAccountSearcher
    """

    opensearch_client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200, 'scheme': 'http'}],  # Explicitly use HTTP
        http_auth=('admin', 'admin'),  # Default credentials for GitHub Actions OpenSearch
        use_ssl=False,
        verify_certs=False,
        ssl_show_warn=False,
    )
    searcher = GameAccountSearcher(opensearch_client)
    
    # Initialize the agent
    agent = AccountRecoveryAgent(
        searcher=searcher,
        min_threshold=5,
        max_iterations=3
    )
    
    # Start recovery session with input from user
    initial_query = input("Enter your query: ")
    # initial_query = "I want to find an account called zacharywallace. i was in an alliance with the word dragon "
    
    # Run the session
    result = agent.run_recovery_session(initial_query)
    
    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final parameters: {result['final_params']}")
    
    print("\nSearch History:")
    for search in result['search_history']:
        print(f"  Iteration {search['iteration']}: {search['hits']} hits ({search['trend']})")

    print(f'\n\n{"@"*50}')
    for result in result["final_results"]["hits"]["hits"]:
        print(f'Results: \n',result["_source"],'\n')

if __name__ == "__main__":
    main()