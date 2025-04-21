# -*- coding: utf-8 -*-
import multiprocessing as mp
from typing import Union

import geopandas as gpd
import pandas as pd
from tqdm import tqdm


def multiprocess(
    df: Union[pd.DataFrame, gpd.GeoDataFrame],
    target_function,
    processes: int = mp.cpu_count(),
    use_pbar: bool = True,
    stepsize=None,
    **kwargs,
):
    """Input every row of the df in the target function. the target function should always contain (idx,row) as the first two arguments
    subsequent arguments are passed with the kwargs.
    Stepsize splits the dataframe into smaller pieces to prevent the loop from breaking.
    !!!!!
    WINDOWS WARNING: DOES NOT WORK IF target_function IS IN THE SAME NOTEBOOK. THIS SHOULD BE DEFINED IN A SEPARATE
    .py FILE THAT IS IMPORTED WHERE YOU CALL THE FUNCTION.
    !!!!!
    """

    if stepsize is None:
        stepsize = len(df)

    def update_pbar(*a):
        """Show progress of the function"""
        pbar.update()

    def main_multi():
        multi_step = {0: 0}
        results_local = {}
        i = 0
        while multi_step[i] != len(df):
            i += 1
            multi_step[i] = min(multi_step[i - 1] + stepsize, len(df))

            # Apply async with or without progressbar
            with mp.Pool(processes=processes) as pool:
                results_local[i] = [
                    pool.apply_async(
                        target_function,
                        args=(idx, row),
                        kwds=kwargs,
                        callback=update_pbar,
                    )
                    for idx, row in df.iloc[multi_step[i - 1] : multi_step[i]].iterrows()
                ]
                results_local[i] = [p.get() for p in results_local[i]]
        return results_local

    # Call target function with or without pbar
    if use_pbar:
        with tqdm(total=len(df), unit="row") as pbar:
            results_local = main_multi()
    else:
        results_local = main_multi()

    # combine results
    results = []
    for i in results_local:
        results.extend(results_local[i])
    return results
