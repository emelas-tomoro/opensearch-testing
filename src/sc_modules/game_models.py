from enum import StrEnum

from pydantic import BaseModel, Field


class Game(BaseModel):
    full_name: str = Field(..., alias="fullName")
    concat_name: str = Field(..., alias="concatName")
    code_name: str = Field(..., alias="codeName")
    helpshift_name: str = Field(..., alias="helpshiftName")

    class Config:
        populate_by_name = True
        frozen = True


MAGIC = Game(
    fullName="Clash of Clans",
    concatName="clashofclans",
    codeName="magic",
    helpshiftName="cc",
)
SCROLL = Game(
    fullName="Clash Royale",
    concatName="clashroyale",
    codeName="scroll",
    helpshiftName="cr",
)
SOIL = Game(
    fullName="Hay Day", concatName="hayday", codeName="soil", helpshiftName="hd"
)
REEF = Game(
    fullName="Boom Beach", concatName="boombeach", codeName="reef", helpshiftName="bb"
)
LASER = Game(
    fullName="Brawl Stars",
    concatName="brawlstars",
    codeName="laser",
    helpshiftName="bs",
)
SQUAD = Game(
    fullName="Squad Busters",
    concatName="squadbusters",
    codeName="squad",
    helpshiftName="sb",
)
ROGUE = Game(fullName="mo.co", concatName="moco", codeName="rogue", helpshiftName="mc")


class GameCode(StrEnum):
    CC = "cc"
    CR = "cr"
    HD = "hd"
    BB = "bb"
    BS = "bs"
    SB = "sb"
    MC = "mc"

    def to_game(self) -> Game:
        """Return the full Pydantic `Game` object that belongs to this code."""
        _LOOKUP = {
            "cc": MAGIC,
            "cr": SCROLL,
            "hd": SOIL,
            "bb": REEF,
            "bs": LASER,
            "sb": SQUAD,
            "mc": ROGUE,
        }
        return _LOOKUP[self.value]
