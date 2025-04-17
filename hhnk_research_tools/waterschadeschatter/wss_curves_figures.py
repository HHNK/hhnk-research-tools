#%%
from hhnk_research_tools.waterschadeschatter.wss_curves_utils import AreaDamageCurveFolders

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib.patches as mpatches

#Globals
STANDAARD_FIGUUR = (10, 10)
LANDGEBRUIK_FIGUUR = (20, 10)
DPI = 300
MAX_PEILVERHOGING = 2.5

# input = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\post\vol_level_curve.csv", index_col=0)
# output_directory = r"C:\temp\HHNK"
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

    def set_x_y_label(self):
        self.xlabel(self.xlabel_dsc)
        self.ylabel(self.ylabel_dsc)
        
    def set_x_y_lim(self):
        self.xlim(self.xlim_min, self.xlim_max) 
        self.ylim(self.ylim_min, self.ylim_max)
        
    def write(self, path, dpi = DPI):
        plt.savefig(path, dpi=dpi)
        plt.close()
        
    
class BergingsCurveFiguur(Figuur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col = 0)
        
        self.xlabel_dsc = "Waterstand (m+NAP)"
        self.ylabel_dsc = "Volume (m3)"
        
    def run(self, output_dir, dpi=DPI):
        for col in self.df.columns:
            valid_data = self.df[col].dropna()
            self.create()
            self.x_y_label()
            self.plot(valid_data)
            self.title(f"bergingscurve voor {col}")
            plotpng = os.path.join(output_dir, f'bergingscurve_{col}.png')
            self.write(plotpng, dpi=dpi)

class PercentageFiguur(Figuur):
    def __init__(self):
        super().__init__(figsize = LANDGEBRUIK_FIGUUR)
        
        self.xlabel_dsc = "Peilverhoging boven streefpeil (m)"
        self.ylabel_dsc = "Percentage t.o.v. totaal"
        self.xlim_min = 0.1
        self.xlim_max = MAX_PEILVERHOGING
        self.ylim_min = 0
        self.ylim_max = 1
        self.x_ticks_list = np.arange(0.1, MAX_PEILVERHOGING+0.1, 0.1)
        self.y_ticks_list =  np.arange(0, 1.1, 0.1)
        self.labels =  ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]

    def set_x_y_ticks(self):
        self.xticks(self.x_ticks) 
        self.yticks(ticks = self.y_ticks_list, labels = self.labels)

    def handles_legend(self, lu_omzetting):
        nieuwe_klasses = np.array(lu_omzetting['nieuwe_klasse'].unique())
        self.color_dict = {}
        colors = []
        self.handels = []
        self.labels = []
        for klasse in nieuwe_klasses:
            label = lu_omzetting.where(lu_omzetting['nieuwe_klasse'] == klasse)['beschrijving'].dropna().unique()[0]
            color = lu_omzetting.where(lu_omzetting['nieuwe_klasse'] == klasse)['kleur'].dropna().unique()[0]
            self.labels.append(label)
            colors.append(color)
        for i in range(0, len(colors)):
            handle = mpatches.Patch(color = colors[i])
            self.handels.append(handle)
            self.color_dict[str(nieuwe_klasses[i])] = colors[i]

    def lu_verdeling_peilgebied(self, id):
        self.df_peilgebied = self.df.loc[self.df['fid'] == id].dropna(axis=1)
        self.df_peilgebied = self.df_peilgebied.drop('fid', axis=1)
        self.df_peilgebied_perc = self.df_peilgebied.divide(self.df_peilgebied.sum(axis=1), axis=0)
        self.df_peilgebied_perc.columns = [str(int(x)) for x in self.df_peilgebied_perc]

        self.lu = []
        self.lu_ids = []
        for col in self.df_peilgebied_perc.columns:
            self.lu.append(self.df_peilgebied_perc[col])
            self.lu_ids.append(col)
    
    def sum_damages(self):
        self.df_sum_damages = self.df.copy()
        self.df_sum_damages['totaal'] = self.df_sum_damages.drop('fid', axis=1).sum(axis=1)

    def plot_schadecurve_totaal(self, id):
        self.ax2 = self.ax.twinx()
        self.ax2.plot(self.df_sum_damages.index.where(self.df['fid'] == id).dropna(), self.df_sum_damages['totaal'].where(self.df_sum_damages['fid'] == id).dropna(), color='black', linewidth=2)
        self.ax2.set_ylabel("Schadebedrag (Euro's)")
        self.ax2.set_ylim(bottom=0)
    
    def combine_classes(self, lu_omzetting, output_path):
        nieuwe_klasses = np.array(lu_omzetting['nieuwe_klasse'].unique())
        samenvoeging_klasses = {'fid' : self.df['fid']} # waar komt self.df vandaan? En misschien een iets duidelijkere naam geven hiervoor.
        for klasse in nieuwe_klasses:
            oude_lu_per_nieuwe_klasse = lu_omzetting.where(lu_omzetting['nieuwe_klasse'] == klasse)['LU_class'].dropna().values.tolist()
            oude_lu_per_nieuwe_klasse = [str(int(x)) for x in oude_lu_per_nieuwe_klasse]
            samenvoeging_klasses[klasse] = self.df.filter(items = oude_lu_per_nieuwe_klasse).sum(axis=1)
        self.df_combined_classes = pd.DataFrame(data = samenvoeging_klasses)
        self.df_combined_classes.to_csv(output_path, sep=',')
        self.df = self.df_combined_classes.copy()

class LandgebruikCurveFiguur(PercentageFiguur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col = 0)
    
    def run(self, lu_omzetting, output_dir, schadecurve_totaal = False, dpi=DPI):
        ids = np.array(self.df['fid'].unique())
        for id in ids:
            self.lu_verdeling_peilgebied(id)
            self.create()
            self.handles_legend(lu_omzetting)
            self.ax.stackplot(self.df_peilgebied_perc.index, self.lu, colors = [self.color_dict.get(x, 'black') for x in self.df_peilgebied_perc.columns])
            if schadecurve_totaal:
                self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.set_x_y_label()
            self.set_x_y_lim()
            self.set_x_y_ticks()
            self.title(f"landgebruikverdeling voor {id}")
    
            self.ax.legend(handles = self.handels, labels = self.labels, bbox_to_anchor=(0.05, -0.05),loc="upper left", ncols=8)

            plotpng = os.path.join(output_dir, f'landgebruikcurve_{id}.png')
            self.write(plotpng, dpi=dpi) 

class DamagesLuCurveFiguur(PercentageFiguur):
    def __init__(self, path):
        super().__init__()
        self.df = pd.read_csv(path, index_col = 0)

    def run(self, lu_omzetting, output_dir, schadecurve_totaal = False, dpi=DPI):
        ids = np.array(self.df['fid'].unique())
        for id in ids:
            self.lu_verdeling_peilgebied(id)

            self.create()
            self.handles_legend(lu_omzetting)
            self.ax.stackplot(self.df_peilgebied_perc.index, self.lu, colors = [self.color_dict.get(x, 'gray') for x in self.df_peilgebied_perc.columns])
            if schadecurve_totaal:
                self.sum_damages()
                self.plot_schadecurve_totaal(id)
            self.x_y_label()
            self.x_y_lim()
            self.x_y_ticks()
            self.title(f"schadeverdeling voor {id}")
    
            self.ax.legend(handles = self.handels, labels = self.labels, bbox_to_anchor=(0.05, -0.05),loc="upper left", ncols=8)

            plotpng = os.path.join(output_dir, f'schadecurve_{id}.png')
            self.write(plotpng, dpi=dpi) 


#%% figuur voor bergingscurve
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



#%%
if __name__ == "__main__":
    pass
    # %% figuur voor lu curve
    # lu_area = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_areas.csv", index_col=0)
    # def_max = 2.5
    # lu_omzetting = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\Project\05 Project execution\Omzettingstabel_landgebruik.csv")

    # def lu_curve(lu_area):
    #     ids = np.array(lu_area['fid'].unique())
    #     os.mkdir(output_directory+"/Landgebruikcurve")
    #     output_dir = output_directory + "/Landgebruikcurve"
    #     for id in ids:
    #         lu_peilgebied = lu_area.loc[lu_area['fid'] == id].dropna(axis=1)
    #         lu_peilgebied = lu_peilgebied.drop('fid', axis=1)
    #         lu_peilgebied_perc = lu_peilgebied.divide(lu_peilgebied.sum(axis=1), axis=0)

    #         columns = []
    #         for col in lu_peilgebied_perc.columns:
    #             columns.append(lu_peilgebied_perc[col])

    #         fig, ax = plt.subplots(figsize=(20,10))    
    #         ax.stackplot(lu_peilgebied_perc.index, columns)
    #         ax.set(xlabel = "Peilverhoging boven streefpeil (m)", 
    #             ylabel = "Percentage t.o.v. totaal",
    #             title = f'Landgebruikcurve {id}',
    #             xlim = (0.1, def_max),
    #             ylim = (0, 1),
    #             xticks = np.arange(0.1, 2.6, 0.1))
    #         ax.set_yticks(ticks = np.arange(0, 1.1, 0.1), labels = ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"])

    #         plotpng = os.path.join(output_dir, f'landgebruikcurve_{id}.png')
    #         plt.savefig(plotpng, dpi = 300)
    #         plt.close()

    # # combineren van klasses landgebruik
    # def reclassify_lu(lu_area, lu_omzetting):
    #     nieuwe_klasses = np.array(lu_omzetting['nieuwe klasse'].unique())
    #     d = {'fid':lu_area['fid']}
    #     samenvoeging_klasses = {}
    #     for klasse in nieuwe_klasses:
    #         oude_lu_per_nieuwe_klasse = lu_omzetting.where(lu_omzetting['nieuwe klasse'] == klasse)['LU class'].dropna().values.tolist()
    #         oude_lu_per_nieuwe_klasse = [str(int(x)) for x in oude_lu_per_nieuwe_klasse]
    #         d[klasse] = lu_area.filter(items = oude_lu_per_nieuwe_klasse).sum(axis=1)
    #     df = pd.DataFrame(data=d)

    #     df.to_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_areas_classes.csv", sep=',')

    # # poging tot labels
    # labeled_land_uses = [False] * len(lu_curve.df_peilgebied_perc.columns)
    # for j, label in enumerate(lu_curve.df_peilgebied_perc.columns):
    #     for i in reversed(range(len(lu_curve.df_peilgebied_perc.index))):
    #         if lu_curve.df_peilgebied_perc.iloc[i, j] > 0.15 and not labeled_land_uses[j]:
    #             y_pos = lu_curve.df_peilgebied_perc.iloc[i, j]-0.1
    #             lu_curve.ax.text(2, y_pos, label, fontsize=8)
    #             labeled_land_uses[j] = True
    #             print(label)
    #             break

    # westzaan = AreaDamageCurveFolders(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan")
    # lu_omzetting = pd.read_csv(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\Project\05 Project execution\Omzettingstabel_landgebruik.csv")
    # #bc = BergingsCurve(westzaan.post.vol_level_curve.path)
    # #bc.run(r"C:\temp\HHNK\Bergingscurve_class")

    # lu_curve = LandgebruikCurve(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_areas_classes.csv")
    # lu_curve.run(lu_omzetting, r"C:\temp\HHNK\Landgebruikcurve_class_classes", schadecurve_totaal=True)

    # # damages = Damages_per_LU_curve(r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_damage.csv")
    # # damages.combine_classes(lu_omzetting, r"C:\Users\benschoj1923\ARCADIS\30225745 - Schadeberekening HHNK - Documenten\External\output\westzaan\output\result_lu_damages_classes.csv")
    # # damages.run(lu_omzetting, r"C:\temp\HHNK\Damages_perc")