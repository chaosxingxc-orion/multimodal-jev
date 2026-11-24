"""Read endpoint scoring settings from online and historical Stage1 configs."""


def scoring_settings(training):
    if 'class_scoring' in training:
        return training['class_scoring']
    return training['epoch_mining']
