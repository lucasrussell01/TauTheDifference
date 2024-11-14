import pandas as pd
import numpy as np

class Selector():

    def __init__(self, logger):
        self.logger = logger

    def select_gen_tau_semilep(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] == 15) & (df['genPartFlav_2'] == 5)] # tau->mu decay and hadronic tau
        n_after = len(df)
        self.logger.debug(f"SemiLeptonic Channel Tau Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_lepton_semilep(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] != 15) & (df['genPartFlav_2'] != 5) & (df['genPartFlav_2'] != 0)] # both prompt electrons/muons
        n_after = len(df)
        self.logger.debug(f"SemiLeptonic Channel Muon Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_jet_semilep(self, df):
        n_bef = len(df)
        df = df[df['genPartFlav_2'] == 0] # hadronic tau is a jet fake
        n_after = len(df)
        self.logger.debug(f"SemiLeptonic Channel Jet Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_tau_hadronic(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] == 5) & (df['genPartFlav_2'] == 5)] # tau->mu decay and hadronic tau
        n_after = len(df)
        self.logger.debug(f"Hadronic Channel Tau Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_lepton_hadronic(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] != 5) & (df['genPartFlav_2'] != 5) & (df['genPartFlav_1'] != 0) & (df['genPartFlav_2'] != 0)]
        n_after = len(df)
        self.logger.debug(f"Hadronic Channel Muon Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_jet_hadronic(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] == 0) | (df['genPartFlav_2'] == 0)] # either hadronic tau is a jet fake
        n_after = len(df)
        self.logger.debug(f"Hadronic Channel Jet Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_id_tt(self, df, sel_cfg):
        n_bef = len(df)
        # VSjet cuts
        df = df[df['idDeepTau2018v2p5VSjet_1'] >= sel_cfg['vsjet_1']]
        df = df[df['idDeepTau2018v2p5VSjet_2'] >= sel_cfg['vsjet_2']]
        # VSe cuts
        df = df[df['idDeepTau2018v2p5VSe_1'] >= sel_cfg['vse_1']]
        df = df[df['idDeepTau2018v2p5VSe_2'] >= sel_cfg['vse_2']]
        # VSmu cuts
        df = df[df['idDeepTau2018v2p5VSmu_1'] >= sel_cfg['vsmu_1']]
        df = df[df['idDeepTau2018v2p5VSmu_2'] >= sel_cfg['vsmu_2']]
        n_after = len(df)
        self.logger.debug(f"DiTau ID Selection: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_id_mt(self, df, sel_cfg):
        n_bef = len(df)
        # Iso cut
        df = df[df['iso_1'] < sel_cfg['iso_1']]
        # DeepTau cuts
        df = df[df['idDeepTau2018v2p5VSjet_2'] >= sel_cfg['vsjet_2']]
        df = df[df['idDeepTau2018v2p5VSe_2'] >= sel_cfg['vse_2']]
        df = df[df['idDeepTau2018v2p5VSmu_2'] >= sel_cfg['vsmu_2']]
        n_after = len(df)
        self.logger.debug(f"MuTau ID Selection: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_id_et(self, df, sel_cfg):
        n_bef = len(df)
        # Iso cut
        df = df[df['iso_1'] < sel_cfg['iso_1']]
        # DeepTau cuts
        df = df[df['idDeepTau2018v2p5VSjet_2'] >= sel_cfg['vsjet_2']]
        df = df[df['idDeepTau2018v2p5VSe_2'] >= sel_cfg['vse_2']]
        df = df[df['idDeepTau2018v2p5VSmu_2'] >= sel_cfg['vsmu_2']]
        n_after = len(df)
        self.logger.debug(f"ETau ID Selection: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_os(self, df, os):
        n_bef = len(df)
        df = df[df['os'] == os]
        n_after = len(df)
        self.logger.debug(f"Opposite sign {os} Selection: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def ditau_trigger_match(self, df, triggers):
        n_bef = len(df)
        if ('trg_doubletau' and 'trg_doubletauandjet') in triggers:
            df = df[((df['trg_doubletau'] == 1) & (df['pt_1'] > 40) & (df['pt_2'] > 40)) |
                    ((df['trg_doubletauandjet'] == 1) & (df['pt_1'] > 35) & (df['pt_2'] > 35) & (df['jpt_1'] > 60))]
        else:
            self.logger.warning("Trigger matching not implemented for channel tt")
        n_after = len(df)
        self.logger.debug(f"DiTau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def mutau_trigger_match(self, df, triggers):
        n_bef = len(df)
        if ('trg_singlemuon') in triggers:
            df = df[(df['trg_singlemuon'] == 1) & (df['pt_1'] > 25)]
        else:
            self.logger.warning("Trigger matching not implemented for channel mt")
        n_after = len(df)
        self.logger.debug(f"MuTau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def etau_trigger_match(self, df, triggers):
        n_bef = len(df)
        if ('trg_singleelectron') in triggers:
            df = df[(df['trg_singleelectron'] == 1) & (df['pt_1'] > 25)]
        else:
            self.logger.warning("Trigger matching not implemented for channel et")
        n_after = len(df)
        self.logger.debug(f"ETau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def cp_weight(self, df):
        cp_weight = 0.5*(df['LHEReweightingWeight_SM'] + df['LHEReweightingWeight_PS'])
        # update central weight
        df['weight'] *= cp_weight
        df['cpweight'] = cp_weight

        # check for large values
        big_cp_weights = df[df['cpweight'] > 10]
        if len(big_cp_weights) > 0:
            self.logger.warning(f"{len(big_cp_weights)} large CP weights found - removing affected events")
            df = df[df['cpweight'] < 10]
            if np.max(big_cp_weights['cpweight']) > 1000:
                self.logger.warning(f"Very large CP weight identified: {np.max(big_cp_weights['cpweight'])}")

        # drop negative weights - combi can be negative (rarely)
        neg_weights = df[df['weight'] < 0]
        if len(neg_weights) > 0:
            self.logger.warning(f"{len(neg_weights)} negative CP weights found - removing affected events")
            df = df[df['weight'] > 0]

        self.logger.debug("CP reweighting applied (negative weights dropped)")
        return df

    def check_sign_weights(self, df):
        neg_weights = df[df['weight'] < 0]
        if len(neg_weights) > 0:
            self.logger.warning(f"{len(neg_weights)} negative weights found - removing affected events")
            df = df[df['weight'] > 0]
        return df

    def mt_cut(self, df):
        # cut mT < 70 GeV
        cut = 70
        n_bef = len(df)
        df = df[df['mt_1'] < cut]
        n_after = len(df)
        self.logger.info(f"Applied mT < {cut}: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df
