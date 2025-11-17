import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title='Quasar and galaxy 2PCFs', layout='wide')

plt.rcParams.update({'axes.linewidth' : 1,
                     'ytick.major.width' : 1,
                     'ytick.minor.width' : 1,
                     'xtick.major.width' : 1,
                     'xtick.minor.width' : 1,
                     'xtick.labelsize': 10, 
                     'ytick.labelsize': 10,
                     'axes.labelsize': 12,
                     'font.family': 'serif',
                     'figure.figsize': (10, 6)
                    })

N_bins = 50
r_edges = np.linspace(1, 20, N_bins+1)
r_avg = (3/4) * (r_edges[1:]**4 - r_edges[:-1]**4) / (r_edges[1:]**3 - r_edges[:-1]**3)

@st.cache_data(show_spinner=False)
def load_combos(filepath):
    with np.load(filepath, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
    
try:
    data = load_combos('Corr_combos_data.npz')
except FileNotFoundError:
    st.error("Could not find 'Corr_combos_data.npz'.")
    st.stop()

def make_plot(sim, z, bhmass_min, fEdd_min, stmass_min, sf):
    mask = (data['sim'] == sim) & (data['z'] == z) & \
        (data['bhmass_min'] == bhmass_min) & (data['fEdd_min'] == fEdd_min) & \
        (data['stmass_min'] == stmass_min) & (data['sf'] == sf)

    if not np.any(mask):
        fig, (ax_xi, ax_R) = plt.subplots(2, 1, figsize=(6,8), sharex=True, constrained_layout=True)
        ax_xi.set_ylabel(r'$\xi$')
        ax_xi.set_title('2PCFs')
        ax_xi.set_xlim(r_edges[0],r_edges[-1])
        ax_xi.set_ylim(0,40)
        ax_R.set_xlabel(r'$r \, [{\rm cMpc} h^{-1}]$')
        ax_R.set_ylabel(r'$R$')
        ax_R.set_title('Bias ratio')
        ax_R.set_ylim(0,2)
        return fig

    else:
        xiqq_mean_arr = data['xiqq'][mask][0]
        xigg_mean_arr = data['xigg'][mask][0]
        xiqg_mean_arr = data['xiqg'][mask][0]
        R_mean_arr = data['R'][mask][0]
        xiqq_sigma_arr = data['xiqq_err'][mask][0]
        xigg_sigma_arr = data['xigg_err'][mask][0]
        xiqg_sigma_arr = data['xiqg_err'][mask][0]
        R_sigma_arr = data['R_err'][mask][0]
        constant = data['constant'][mask][0]
        constant_err = data['constant_err'][mask][0]

        fig, (ax_xi, ax_R) = plt.subplots(2, 1, figsize=(6,8), sharex=True, constrained_layout=True)
        
        ax_xi.errorbar(r_avg, xiqq_mean_arr, yerr=xiqq_sigma_arr, \
                       fmt='o-', markersize=3.5, lw=1, elinewidth=2.5, capsize=3, alpha=0.75, \
                       color='xkcd:royal blue', label=r'$\xi_{qq}$')
        ax_xi.errorbar(r_avg, xigg_mean_arr, yerr=xigg_sigma_arr, \
                       fmt='o-', markersize=3.5, lw=1, elinewidth=2.5, capsize=3, alpha=0.75, \
                       color='xkcd:turquoise', label=r'$\xi_{gg}$')
        ax_xi.errorbar(r_avg, xiqg_mean_arr, yerr=xiqg_sigma_arr, \
                       fmt='o-', markersize=3.5, lw=1, elinewidth=2.5, capsize=3, alpha=0.75, \
                       color='xkcd:magenta', label=r'$\xi_{qg}$')
        ax_xi.legend(loc='upper right')
        ax_xi.set_ylabel(r'$\xi$')
        ax_xi.set_title('2PCFs')
        ax_xi.set_xlim(r_edges[0],r_edges[-1])
        ax_xi.set_ylim(0,40)
        
        ax_R.errorbar(r_avg, R_mean_arr, yerr=R_sigma_arr, \
                      fmt='o-', markersize=3.5, lw=1, elinewidth=2.5, capsize=3, alpha=0.75, color='xkcd:royal blue')
        ax_R.axhline(constant, ls='dashed', lw=1.5, color='xkcd:magenta', \
                     label=f'Best fit: $R = {np.round(constant,3)} \\pm {np.round(constant_err,3)}$')
        ax_R.fill_between([r_edges[0], r_edges[-1]], constant-constant_err, constant+constant_err, \
                          alpha=0.45, color='xkcd:magenta')
        ax_R.legend(loc='lower center')
        ax_R.set_xlabel(r'$r \, \, [{\rm cMpc} \, h^{-1}]$')
        ax_R.set_ylabel(r'$R = b_q / b_g$')
        ax_R.set_title('Bias ratio')
        ax_R.set_ylim(0,2)
        
        return fig

sim = st.sidebar.selectbox('Simulation', options=['TNG', 'ASTRID'], index=0)

z = st.sidebar.select_slider('$z$', options=[1, 2, 3], value=1)

bhmass_labels = ['7', '7.5', '8', '8.5', '9']
bhmass_values = [10**(7), 10**(7.5), 10**(8), 10**(8.5), 10**(9)]
bhmass_map = dict(zip(bhmass_labels, bhmass_values))
bhmass_slider = st.sidebar.select_slider(r'$\log_{10}{(M_{BH,min}/{\rm M}_\odot)}$', options=bhmass_labels, value='7')
bhmass_min = bhmass_map[bhmass_slider]

fEdd_labels = ['No min', '0.001', '0.01', '0.05', '0.1']
fEdd_values = [-1, 0.001, 0.01, 0.05, 0.1]
fEdd_map = dict(zip(fEdd_labels, fEdd_values))
fEdd_slider = st.sidebar.select_slider(r'$f_{Edd,min}$', options=fEdd_labels, value='No min')
fEdd_min = fEdd_map[fEdd_slider]

stmass_labels = ['10', '10.25', '10.5', '10.75', '11']
stmass_values = [10**(10), 10**(10.25), 10**(10.5), 10**(10.75), 10**(11)]
stmass_map = dict(zip(stmass_labels, stmass_values))
stmass_slider = st.sidebar.select_slider(r'$\log_{10}{(M_{\star,min}/{\rm M}_\odot)}$', options=stmass_labels, value='10')
stmass_min = stmass_map[stmass_slider]

sf_labels = ['Star-forming', 'Quiescent']
sf_values = [True, False]
sf_map = dict(zip(sf_labels, sf_values))
sf_box = st.sidebar.selectbox('Galaxy type', options=sf_labels, index=0)
sf = sf_map[sf_box]

fig = make_plot(sim, z, bhmass_min, fEdd_min, stmass_min, sf)

st.pyplot(fig, clear_figure=True)