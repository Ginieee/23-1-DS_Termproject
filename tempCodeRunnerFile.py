
# # Apply PCA to each dataframe
# for i, df in enumerate(df_list):
#     # Extract the numerical columns from the dataframe
#     numeric_cols = df.select_dtypes(include=[float, int])
    
#     # Perform PCA
#     pca = PCA(n_components=5)  # Specify the desired number of components
#     transformed_data = pca.fit_transform(numeric_cols)
    
#     # Create a new dataframe with the transformed data
#     transformed_df = pd.DataFrame(transformed_data, columns=['PC1', 'PC2'])
    
#     # Update the dataframe in the df_list
#     df_list[i] = tra