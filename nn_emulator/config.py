# config.py

#Configuration for redshifts and pipeline
#Arguments in order for passing into get_train_test_data function in data_processing.py
redshift_configs = {
    4.2: {
        "base_path": "/data/og313/lyalpha/code_wdm/analysis_repatchy",
        "in_file": "inputs4.2_repatchy.txt",
        "out_file": "ps4.2_repatchy.txt",
        "hidden": 120,
        "learning_rate": 0.0001,
        "batch_size": 18,
        "patience": 500,
        "patience_scheduler": 30,
        "epochs": 550
    },
    4.6: {
        "base_path": "/data/og313/lyalpha/code_wdm/analysis_repatchy",
        "in_file": "inputs4.6_repatchy.txt",
        "out_file": "ps4.6_repatchy.txt",
        "hidden": 120,
        "learning_rate": 0.0001 ,
        "batch_size": 18,
        "patience": 500,
        "patience_scheduler": 40,
        "epochs": 550
    },
    5.0: {
        "base_path": "/data/og313/lyalpha/code_wdm/analysis_repatchy",
        "in_file": "inputs5.0_repatchy.txt",
        "out_file": "ps5.0_repatchy.txt",
        "hidden": 120,
        "learning_rate": 0.0001 ,
        "batch_size": 18,
        "patience": 500,
        "patience_scheduler": 40,
        "epochs": 550
    }
}
