class Experiment(object):
    pass

expts = dict()

expts["vae-self"] = Experiment()
expts["vae-self"].gen_from_self    = True
expts["vae-self"].gen_from_context = False
expts["vae-self"].gen_from_txr     = False
expts["vae-self"].gen_from_lstm    = False
expts["vae-self"].reparametrize_merged = False
expts["vae-self"].reparametrize_lm     = False

expts["vae-ctx"] = Experiment()
expts["vae-ctx"].gen_from_self    = True
expts["vae-ctx"].gen_from_context = True
expts["vae-ctx"].gen_from_txr     = False
expts["vae-ctx"].gen_from_lstm    = False
expts["vae-ctx"].reparametrize_merged = True
expts["vae-ctx"].reparametrize_lm     = False

expts["lstm"] = Experiment()
expts["lstm"].gen_from_self    = True
expts["lstm"].gen_from_context = False
expts["lstm"].gen_from_txr     = False
expts["lstm"].gen_from_lstm    = True
expts["lstm"].reparametrize_merged = False
expts["lstm"].reparametrize_lm     = True

expts["txr"] = Experiment()
expts["txr"].gen_from_self    = True
expts["txr"].gen_from_context = False
expts["txr"].gen_from_txr     = True
expts["txr"].gen_from_lstm    = False
expts["txr"].reparametrize_merged = False
expts["txr"].reparametrize_lm     = True
