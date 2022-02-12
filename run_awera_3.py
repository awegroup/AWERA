import os
from AWERA import config, ChainAWERA

training_settings = []
prediction_settings = []

# Test run on 5 special locations
for j in range(5):
    settings = {
        'Processing': {'n_cores': 10},
        'Data': {
            'n_locs': 1,
            'location_type': 'europe_ref_0'},
        'Clustering': {
            'training': {
                'n_locs': 1,
                'location_type': 'europe_ref_{}'.format(j)
                }
            },
        }
    training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 10},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'n_clusters': 8,
        'training': {
            'n_locs': 5,
            'location_type': 'europe_ref'
            }
        },
    }
training_settings.append(settings)

# multiple locations
settings = {
    'Processing': {'n_cores': 10},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 50,
            'location_type': 'europe'
            }
        },
    }

training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 10},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 200,
            'location_type': 'europe'
            }
        },
    }
training_settings.append(settings)

# Multiple locations plus 5 special locs
settings = {
    'Processing': {'n_cores': 10},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 55,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 10},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 205,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)


# very many locations
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 5000,
            'location_type': 'europe'
            }
        },
    }
training_settings.append(settings)
# TODO include prediction
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 1000,
            'location_type': 'europe'
            }
        },
    }
training_settings.append(settings)
settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 5005,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)

settings = {
    'Processing': {'n_cores': 23},
    'Data': {'n_locs': 1,
             'location_type': 'europe_ref_0'},
    'Clustering': {
        'training': {
            'n_locs': 1005,
            'location_type': 'europe_incl_ref'
            }
        },
    }
training_settings.append(settings)

data_settings = [
    {'n_locs': 1,
     'location_type': 'europe_ref_0'},
    {'n_locs': 1,
     'location_type': 'europe_ref_1'},
    {'n_locs': 1,
     'location_type': 'europe_ref_2'},
    {'n_locs': 1,
     'location_type': 'europe_ref_3'},
    {'n_locs': 1,
     'location_type': 'europe_ref_4'},
    {'n_locs': 5,
     'location_type': 'europe_ref'}
    ]
# TODO make clear how each module runs with the inout of the previous
# power_model.load() .generate() how is the object made/passed


if __name__ == '__main__':

    prediction_settings = []
    for settings in training_settings:
        prediction_settings.append(settings['Clustering'])

    from AWERA.validation.validation import ValidationChain
    # prediction using data reference
    # TODO rn for all ref locs
    i_data = 5
    data_settings = data_settings[i_data]
    # Update starting settings to config
    config.update({'Data': data_settings,
                   'Clustering': prediction_settings[0]})

    val_chain = ValidationChain(config)

    val_chain.aep_vs_n_locs(
        prediction_settings=prediction_settings,
        data_settings=data_settings,
        # Training settings organised in the same way as the data settings
        i_ref=i_data,
        set_labels=None
        )

    #plt.show()

