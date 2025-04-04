#%%
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#Globals
STANDAARD_FIGUUR = (10, 10)
LANDGEBRUIK_FIGUUR = (20, 10)
DPI = 300
MAX_PEILVERHOGING = 2.5

input = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\post\vol_level_curve.csv", index_col=0)
output_directory = r"C:\temp\HHNK"
class Figuur:
    def __init__(self, figsize=STANDAARD_FIGUUR):
        self.figsize = figsize
    
    def create(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
    
    def plot(self, dataframe):
        self.ax.plot(dataframe)

    def xlabel(self, naam):
        self.ax.set(xlabel = naam)

    def ylabel(self, naam):
        self.ax.set(ylabel = naam)

    def title(self, naam):
        self.ax.set(title = naam)

    def xlim(self, xmin, xmax):
        self.ax.set(xlim = (xmin, xmax))

    def ylim(self, ymin, ymax):
        self.ax.set(ylim = (ymin, ymax))
    
    def xticks(self, ticks, *labels):
        self.ax.set_xticks(ticks, *labels)
    
    def yticks(self, ticks, labels=[]):
        self.ax.set_yticks(ticks, labels)

    def write(self, path, dpi = DPI):
        plt.savefig(path, dpi=dpi)
        plt.close()

class BergingsCurve(Figuur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col = 0)

    def x_y_label(self):
        self.xlabel("Waterstand (m+NAP)")
        self.ylabel("Volume (m3)")
        
    def run(self, output_dir, dpi=DPI):
        for col in self.df.columns:
            valid_data = self.df[col].dropna()
            self.create()
            self.x_y_label()
            self.plot(valid_data)
            self.title(f"bergingscurve voor {col}")
            plotpng = os.path.join(output_dir, f'bergingscurve_{col}.png')
            self.write(plotpng, dpi=dpi)

class LandgebruikCurve(Figuur):
    def __init__(self, path):
        super().__init__(figsize = LANDGEBRUIK_FIGUUR)
        self.df = pd.read_csv(path, index_col = 0)

    def x_y_label(self):
        self.xlabel("Peilverhoging boven streefpeil (m)")
        self.ylabel("Percentage t.o.v. totaal")
    
    def x_y_lim(self):
        self.xlim(0.1, MAX_PEILVERHOGING)
        self.ylim(0, 1)
    
    def x_y_ticks(self):
        self.xticks(np.arange(0.1, MAX_PEILVERHOGING+0.1, 0.1))
        self.yticks(ticks = np.arange(0, 1.1, 0.1), labels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])

    def per_peilgebied(self, id):
        self.df_peilgebied = self.df.loc[lu_area['fid'] == id].dropna(axis=1)
        self.df_peilgebied = self.df_peilgebied.drop('fid', axis=1)
        self.df_peilgebied_perc = self.df_peilgebied.divide(self.df_peilgebied.sum(axis=1), axis=0)

    def lu_peilgebied(self):
        self.lu = []
        self.lu_ids = []
        for col in self.df_peilgebied_perc.columns:
            self.lu.append(self.df_peilgebied_perc[col])
            self.lu_ids.append(col)


    def run(self, output_dir, dpi=DPI):
        ids = np.array(self.df['fid'].unique())
        for id in ids:
            self.per_peilgebied(id)
            self.lu_peilgebied()

            self.create()
            self.ax.stackplot(self.df_peilgebied_perc.index, self.lu, labels = self.lu_ids)
            self.x_y_label()
            self.x_y_lim()
            self.x_y_ticks()

            plotpng = os.path.join(output_dir, f'landgebruikcurve_{id}.png')
            self.write(plotpng, dpi=dpi) 
    
#%%
def bergingscurve(input, output_directory):
    os.mkdir(output_directory+"/Bergingscurve")
    output_dir = output_directory + "/Bergingscurve"
    for col in input.columns:
        valid = input[col].dropna()
        # Prettier plotting with seaborn
        sns.set(font_scale=1.5)
        sns.set_style("whitegrid")
        
        # Format histograms
        plt.rcParams['figure.figsize'] = (8, 8)
         
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.lineplot(valid)
        ax.set(xlabel = 'Waterstand (m+NAP)', 
            ylabel = 'Volume (m3)',
            title = f'bergingscurve voor {col}',
            )
        
        plotpng = os.path.join(output_dir, f'bergingscurve_{col}.png')
        plt.savefig(plotpng, dpi = 300)
        plt.close()

# %%
lu_area = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_areas.csv", index_col=0)
def_max = 2.5

def lu_curve(lu_area):
    ids = np.array(lu_area['fid'].unique())
    os.mkdir(output_directory+"/Landgebruikcurve")
    output_dir = output_directory + "/Landgebruikcurve"
    for id in ids:
        lu_peilgebied = lu_area.loc[lu_area['fid'] == id].dropna(axis=1)
        lu_peilgebied = lu_peilgebied.drop('fid', axis=1)
        lu_peilgebied_perc = lu_peilgebied.divide(lu_peilgebied.sum(axis=1), axis=0)

        columns = []
        for col in lu_peilgebied_perc.columns:
            columns.append(lu_peilgebied_perc[col])

        fig, ax = plt.subplots(figsize=(20,10))    
        ax.stackplot(lu_peilgebied_perc.index, columns)
        ax.set(xlabel = "Peilverhoging boven streefpeil (m)", 
            ylabel = "Percentage t.o.v. totaal",
            title = f'Landgebruikcurve {id}',
            xlim = (0.1, def_max),
            ylim = (0, 1),
            xticks = np.arange(0.1, 2.6, 0.1))
        ax.set_yticks(ticks = np.arange(0, 1.1, 0.1), labels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])

        plotpng = os.path.join(output_dir, f'landgebruikcurve_{id}.png')
        plt.savefig(plotpng, dpi = 300)
        plt.close()

#%%
if __name__ == "__main__":
    westzaan = AreaDamageCurveFolders(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan")
    #bc = BergingsCurve(westzaan.post.vol_level_curve.path)
    #bc.run(r"C:\temp\HHNK\Bergingscurve_class")

    luc = LandgebruikCurve(westzaan.output.result_lu_areas.path)
    luc.run(r"C:\temp\HHNK\Landgebruikcurve_class")