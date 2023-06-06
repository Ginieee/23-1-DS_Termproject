
# setting(df_list)
# addDirtyData(df_list, df_name_list)

# # ==============================================
# # 4. Data Exploration
# # ==============================================
# print('Data Exploration ----------------------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# name_list = ["Garlic", "Napa Cabbage", "Radish", "Pepper"]

# setting_exploration(df_list)
# data_exploration(df_list, name_list)

# # ==============================================
# # 5. Remove Dirty Data
# # ==============================================
# print('Remove Dirty Data ---------------------------------------------------------------------------------------------------------------------------------------------------------')
# garlic_df = pd.read_csv("add_dirtydata/garlic_df.csv", low_memory=False)
# napa_cabbage_df = pd.read_csv("add_dirtydata/napa_cabbage_df.csv", low_memory=False)
# radish_df = pd.read_csv("add_dirtydata/radish_df.csv", low_memory=False)
# pepper_df = pd.read_csv("add_dirtydata/pepper_df.csv", low_memory=False)

# df_list = [garlic_df, napa_cabbage_df, radish_df, pepper_df]
# df_name_list = ["garlic_df", "napa_cabbage_df", "radish_df", "pepper_df"]
# item_list = ['마늘', '배추', '무', '건고추']
# file_path = "remove_dirtyData/"

# after_remove_dirty_data_list = remove_save(df_list.copy(), df_name_list, item_list, file_path)
# # compare_dirty_data(df_list, after_remove_dirty_data_list, item_list)
