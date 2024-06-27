import pandas as pd
from ftlangdetect import detect


def preprocesstext(df: pd.DataFrame) -> pd.DataFrame:
    newdf = replace_idem_with_description(df)
    newdf = remove_lang_description(newdf)
    newdf = remove_multiple_intervals_description(newdf)
    return newdf


def replace_idem_with_description(df: pd.DataFrame) -> pd.DataFrame:
    # Find closest row index before rownr that does not start with "idem"
    def find_previous_nonidem(rownr: int) -> int:
        currow = rownr
        while df.iloc[currow]["beschrijving"].lower().startswith("idem"):
            currow -= 1
        assert all(df.iloc[rownr][["x", "y"]] == df.iloc[currow][["x", "y"]])
        return currow  # , df.iloc[currow]['beschrijving']

    # For observations starting with "idem" in the "beschrijving" replace "idem" with the "beschrijving" at the previous line
    idemind = df["beschrijving"].str.lower().str.startswith("idem")
    if idemind.sum() == 0:
        return df
    replaceind = [*map(find_previous_nonidem, idemind[idemind].index.tolist())]
    temp = df.iloc[replaceind][["beschrijving"]].reset_index(drop=True) + df[idemind][
        "beschrijving"
    ].str.partition("idem").rename(columns={2: "beschrijving"})[
        ["beschrijving"]
    ].reset_index(drop=True)
    temp = temp.set_index(df.loc[idemind, "beschrijving"].index)
    newdf = df.copy(deep=True)
    newdf.loc[idemind, "beschrijving"] = temp
    return newdf


def remove_lang_description(
    df: pd.DataFrame, lang: str = "fr", descr_col: str = "beschrijving"
) -> pd.DataFrame:
    """Filter out descriptions in selected language"""
    langid = df[descr_col].apply(lambda v: detect(text=v, low_memory=False)["lang"])
    newdf = df[langid != lang]
    return newdf


def remove_multiple_intervals_description(
    df: pd.DataFrame, descr_col: str = "beschrijving"
) -> pd.DataFrame:
    """Drop observations where the description describes different intervals in one entry"""
    return df.drop(
        index=df.loc[
            df[descr_col].str.match(
                "van [0-9]+ tot [0-9]+ .* van .* tot .*|(^|KERNSTROOK [0-9]*|Kern)[ ]?[0-9\.\,]+[ m]?([- ]+|tot )[0-9\.\,]+[ ]?m",
                case=False,
            )
        ].index
    )
