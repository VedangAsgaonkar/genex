
def get_model(sess, hps):
    if hps.model == 'acflow_classifier':
        from .acflow_classifier import Model
        model = Model(sess, hps)
    elif hps.model == 'acflow_regressor':
        from .acflow_regressor import Model
        model = Model(sess, hps)
    else:
        raise Exception()

    return model