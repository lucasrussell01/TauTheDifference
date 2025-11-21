import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
plt.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 14})
plt.tight_layout()

class stacked_histogram:
    
    '''
    Plot stacked histograms for different contributions
    '''
    
    def __init__(self, var_name, ax, bins):
        self.var_name = var_name
        self.ax = ax
        self.bins = bins
        self.bin_centre = bins[:-1]+ np.diff(bins)/2
        self.step_edges = np.append(bins,2*bins[-1]-bins[-2]) # for outline

        self.bin_centre = bins[:-1]+ np.diff(bins)/2
        self.step_edges = np.append(bins,2*bins[-1]-bins[-2]) # for outline

        self.bottom_bar = 0 # where to plot from (stacking MC contributions)
        self.bottom_step = 0 # where to plot from (outline)
        # colour palette
        self.colours = {
            'yellow': (243/255,170/255,37/255),
            'purple': (152/255, 152/255, 201/255),
            'red': (213/255, 88/255, 84/255),
            'blue': (47/255, 153/255, 205/255),
            'pink': (241/255, 204/255, 225/255),
            'brown': (193/255, 131/255, 119/255),
            'light_green': (159/255, 223/255, 132/255),
            'dark_green': (0/255, 153/255, 0/255),
            'light_yellow': (247/255, 243/255, 154/255),
            'orange': (206/255, 104/255, 50/255),
            'blue_line': (2/255, 114/255, 187/255),
            'red_line': 'darkblue', #(203/255, 68/255, 10/255),
            'green_line': (0/255, 153/255, 0/255),
            'orange_line': "#ff5e02"
        }
        # Dictionary to store process information from genuine and fake backgrounds
        self.bkg_process_info = {
            # Genuine Tau Processes
            "DY_tau": {'color': self.colours['yellow'], 'label': r'$Z\to\tau\tau$ (genuine)'},
            "Top_tau": {'color': self.colours['light_yellow'], 'label': r'$t\bar{t}$/single $t$ (genuine)'},
            "VV_tau": {'color': self.colours['orange'], 'label': r'Diboson (genuine)'},
            "Top_NJ": {'color': self.colours['light_yellow'], 'label': r'$t\bar{t}$ (non jet)'},
            "VV_NJ": {'color': self.colours['light_yellow'], 'label': r'Diboson/single $t$ (non jet)'},
            # Lepton Processes
            "DY_lep": {'color': self.colours['blue'], 'label': r'$Z\to \ell\ell$'},
            # Jet Processes
            "WJets": {'color': self.colours['red'], 'label': 'W+jets'},
            "QCD": {'color': self.colours['pink'], 'label': r'Same Sign Data'},
            "Top_jet": {'color': self.colours['purple'], 'label': r'$t\bar{t}$/single $t$ '},
            "DY_jet": {'color': self.colours['dark_green'], 'label': r'$Z\to \tau\tau$ (jet $\to \tau_h$)'},
            "VV_jet": {'color': self.colours['brown'], 'label': r'Diboson (jet $\to \tau_h$)'},
            # General Processes
            "Jet_Fakes": {'color': self.colours['light_green'], 'label': r'jet $\to \tau_h$'},
            "DY": {'color': self.colours['yellow'], 'label': r'$Z\to\tau\tau$'},
            "EW": {'color': self.colours['red'], 'label': r'Electroweak'},
            "OtherGenuine": {'color': self.colours['light_yellow'], 'label': r'Other genuine $\tau_h$'},
            "OtherFake": {'color': self.colours['light_green'], 'label': r'Other jet $\to \tau_h$'}
        }
        self.signal_process_info = {
            # ggH
            "ggH": {'color': self.colours['red_line'], 'label': r'ggH$\to\tau\tau$'},
            # VBF
            "VBF": {'color': self.colours['blue_line'], 'label': r'qqH$\to\tau\tau$'},
            # VH
            "VH": {'color': self.colours['green_line'], 'label': r'VH$\to\tau\tau$'}
        }
        self.signal_numbers = np.zeros(len(self.bins)-1) # track N signal events in each bin
        self.bkg_numbers = np.zeros(len(self.bins)-1) # track N bkg events in each bin
        

    def add_bkg(self, df, process_name, weight='weight'):
        # This function add a process to the histogram
        # Plot a histogram for the process
        counts = np.histogram(df[self.var_name], bins=self.bins, weights=df[weight])[0]
        self.ax.bar(self.bin_centre, counts, width = np.diff(self.bins), bottom = self.bottom_bar,
               color = self.bkg_process_info[process_name]['color'], label = self.bkg_process_info[process_name]['label'])
        # Plot an outline
        steps = np.append(np.insert(counts,0,0.0),0.0)
        self.ax.step(self.step_edges, steps + self.bottom_step, color='black', linewidth = 0.5)
        # Update the bottom for the next process
        self.bottom_bar += counts
        self.bottom_step += steps
        self.bkg_numbers += np.histogram(df[self.var_name], bins=self.bins, weights=df[weight])[0]
        return self.ax

    def add_signal(self, df, process_name, weight='weight'):
        # This function adds signal to the histogram
        self.ax.hist(df[self.var_name], bins=self.bins, weights=df[weight], histtype="step",
                     color = self.signal_process_info[process_name]['color'], linewidth = 2, label = self.signal_process_info[process_name]['label'])
        self.signal_numbers += np.histogram(df[self.var_name], bins=self.bins, weights=df[weight])[0]
        return self.ax

    def add_total_bkg(self):
        # Add a thick black line on top of the total background
        self.ax.step(self.step_edges, self.bottom_step, color='black', linewidth = 2, label = 'Total Background')
        return self.ax

    def get_ax(self, xlabel=None, lumi= 60.90, ncol=1, fontsmall=True):
        # Axes
        if xlabel is not None:
            self.ax.set_xlabel(rf"{xlabel}")
        else:
            self.ax.set_xlabel(rf"{self.var_name}")
        self.ax.set_ylabel(f"Weighted Events")
        # CMS style
        self.ax.text(0.6, 1.02, fr"{(lumi):.2f} fb$^{{-1}}$ (13.6 TeV)", fontsize=14, transform=self.ax.transAxes)
        self.ax.text(0.01, 1.02, 'CMS', fontsize=18, transform=self.ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        self.ax.text(0.14, 1.02, 'Work in Progress', fontsize=14, transform=self.ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
        # legend
        if fontsmall:
            self.ax.legend(frameon=1, framealpha=1, loc='upper left', ncol=ncol, fontsize=10)
        else:
            self.ax.legend(frameon=1, framealpha=1, loc='upper left', ncol=ncol, fontsize=14)
        return self.ax

    def get_counts(self):
        return self.signal_numbers, self.bkg_numbers

    def get_max(self):
        return np.max(self.bottom_bar) # return the maximum value of the stacked histogram

