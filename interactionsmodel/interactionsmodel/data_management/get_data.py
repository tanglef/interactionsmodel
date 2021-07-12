import os
import tarfile
import fnmatch
import pandas as pd
from download import download


def download_data(dir_to_data, all_regulatory_regions=True, return_X_Y_col_names=False):
    if not os.path.exists(dir_to_data):
        print("Making the directory")
        os.makedirs(dir_to_data)
    if all_regulatory_regions:
        df_name = "data_abs_trans_log_short0_sb_DNCORE.txt"
        if fnmatch.filter(os.listdir(dir_to_data), df_name):
            if return_X_Y_col_names:
                data = pd.read_table(os.path.join(dir_to_data, df_name), sep=r"\s+")
                data_numpy = data.to_numpy()
                Y = data_numpy[:, 0:241]
                X = data_numpy[:, 241:401]
                colnames = data.columns.values
                Y_col_names = colnames[0:241]
                X_col_names = colnames[241:401]
                return (X, Y, X_col_names, Y_col_names)
            else:
                data = pd.read_table(
                    os.path.join(dir_to_data, df_name), sep=r"\s+"
                ).to_numpy()
                Y = data[:, 0:241]
                X = data[:, 241:401]
                return (X, Y)
        else:
            #  download data from Sophie WebSite
            print("Download Data")
            url = (
                "https://upvdrive.univ-montp3.fr/index.php/"
                + "s/S3sfnJLdy5aeYmQ/download"
            )
            path_target = os.path.join(dir_to_data, df_name)
            download(url, path_target)

            if return_X_Y_col_names:
                data = pd.read_table(dir_to_data + df_name, sep=r"\s+")
                data_numpy = data.to_numpy()
                Y = data_numpy[:, 0:241]
                X = data_numpy[:, 241:401]
                colnames = data.columns.values
                Y_col_names = colnames[0:241]
                X_col_names = colnames[241:401]
                return (X, Y, X_col_names, Y_col_names)
            else:
                data = pd.read_table(dir_to_data + df_name, sep=r"\s+").to_numpy()
                Y = data[:, 0:241]
                X = data[:, 241:401]
                return (X, Y)

    else:
        df_name = "genes_promoter_predictive_variables.txt"
        y_df_name = "expression_data_genes_log_241samples_TCGA.txt"
        if fnmatch.filter(os.listdir(dir_to_data), df_name):
            X = pd.read_csv(os.path.join(dir_to_data, df_name), sep=r"\s+").to_numpy()
            Y = pd.read_csv(os.path.join(dir_to_data, y_df_name), sep=r"\s+").to_numpy()
            return (X, Y)
        else:
            #  download data from Sophie WebSite
            print("Download Data")
            url = (
                "http://www.univ-montp3.fr/miap/~lebre/IBCdata/"
                + "genes_data_predicted_and_predictive_variables.tar"
            )
            df_name_upload = "Computed_for_all_genes_data.tar"
            path_target = os.path.join(dir_to_data, df_name_upload)
            download(url, path_target)
            tar_files = tarfile.open(path_target)
            tar_files.extractall(path=dir_to_data)

            X = pd.read_csv(os.path.join(dir_to_data, df_name), sep=r"\s+").to_numpy()
            Y = pd.read_csv(os.path.join(dir_to_data, y_df_name), sep=r"\s+").to_numpy()
            return (X, Y)
