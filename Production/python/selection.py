import pandas as pd

class Selector():

    def __init__(self, logger):
        self.logger = logger

    def select_gen_tau(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] == 15) & (df['genPartFlav_2'] == 5)] # tau->mu decay and hadronic tau
        n_after = len(df)
        self.logger.debug(f"Tau Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_lepton(self, df):
        n_bef = len(df)
        df = df[(df['genPartFlav_1'] != 15) & (df['genPartFlav_2'] != 5) & (df['genPartFlav_2'] != 0)] # both prompt muons
        n_after = len(df)
        self.logger.debug(f"Muon Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def select_gen_jet(self, df):
        n_bef = len(df)
        df = df[df['genPartFlav_2'] == 0] # hadronic tau is a jet fake
        n_after = len(df)
        self.logger.debug(f"Jet Gen Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
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
            logger.warning("Trigger matching not implemented for channel tt")
        n_after = len(df)
        self.logger.debug(f"DiTau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def mutau_trigger_match(self, df, triggers):
        n_bef = len(df)
        if ('trg_singlemuon') in triggers:
            df = df[(df['trg_singlemuon'] == 1) & (df['pt_1'] > 25)]
        else:
            logger.warning("Trigger matching not implemented for channel mt")
        n_after = len(df)
        self.logger.debug(f"MuTau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def etau_trigger_match(self, df, triggers):
        n_bef = len(df)
        logger.warning("Trigger matching not implemented for channel et")
        n_after = len(df)
        self.logger.debug(f"ETau Trigger Matching: {(n_after/n_bef)*100:.2f}% kept- {n_after} events remaining")
        return df

    def cp_weight(self, df):
        cp_weight = 0.5*(df['LHEReweightingWeight_SM'] + df['LHEReweightingWeight_PS'])
        df['weight'] *= cp_weight
        # drop negative weights - combi can be negative (rarely)
        df = df[df['weight'] > 0]
        self.logger.debug("CP reweighting applied (negative weights dropped)")
        return df

