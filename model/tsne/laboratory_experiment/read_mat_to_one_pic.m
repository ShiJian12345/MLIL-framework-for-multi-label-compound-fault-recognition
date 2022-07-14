
% fold_path = "G:\B328_SERVER\TP\coder\paper\duanjie_4speed_3type_2load\DRL_4_model_duanjie_4_3_2\results\1\pre_trained\tsne_files\tsne_mats\"
fold_path = ".\tsne_mats\"
label=cell2mat(struct2cell(load(fold_path+"label.mat")))


raw_data_dri_2=cell2mat(struct2cell(load(fold_path+"ML_CNN_1_raw_test_data_2_dimension.mat")))
raw_data_fan_2=cell2mat(struct2cell(load(fold_path+"ML_CNN_2_raw_test_data_2_dimension.mat")))
raw_data_dri_fan_2=cell2mat(struct2cell(load(fold_path+"MSIL_DARL_raw_test_data_2_dimension.mat")))
ML_CNN_1_2=cell2mat(struct2cell(load(fold_path+"ML_CNN_1_fc1_2_dim.mat")))
ML_CNN_2_2=cell2mat(struct2cell(load(fold_path+"ML_CNN_2_fc1_2_dim.mat")))
MSIF_DARL_2=cell2mat(struct2cell(load(fold_path+"MSIL_DARL_fc1_1step_2_dim.mat")))
MSIF_ML_CNN_2=cell2mat(struct2cell(load(fold_path+"MSIL_ML_CNN_fc1_2_dim.mat")))


raw_data_dri_3=cell2mat(struct2cell(load(fold_path+"ML_CNN_1_raw_test_data_3_dimension.mat")))
raw_data_fan_3=cell2mat(struct2cell(load(fold_path+"ML_CNN_2_raw_test_data_3_dimension.mat")))
raw_data_dri_fan_3=cell2mat(struct2cell(load(fold_path+"MSIL_DARL_raw_test_data_3_dimension.mat")))
ML_CNN_1_3=cell2mat(struct2cell(load(fold_path+"ML_CNN_1_fc1_3_dim.mat")))
ML_CNN_2_3=cell2mat(struct2cell(load(fold_path+"ML_CNN_2_fc1_3_dim.mat")))
MSIF_DARL_3=cell2mat(struct2cell(load(fold_path+"MSIL_DARL_fc1_1step_3_dim.mat")))
MSIF_ML_CNN_3=cell2mat(struct2cell(load(fold_path+"MSIL_ML_CNN_fc1_3_dim.mat")))


% 
%plot_function(raw_data2, label, 2, "raw_test_data_2_dimension")
cell_matrix_2_dimension={raw_data_dri_2,raw_data_fan_2,raw_data_dri_fan_2,ML_CNN_1_2,...
    ML_CNN_2_2,MSIF_DARL_2,MSIF_ML_CNN_2};
cell_matrix_3_dimension={raw_data_dri_3,raw_data_fan_3,raw_data_dri_fan_3,ML_CNN_1_3,...
    ML_CNN_2_3,MSIF_DARL_3,MSIF_ML_CNN_3};
cell_matrix_list={cell_matrix_2_dimension,cell_matrix_3_dimension};


cell_file_name_2_dimension={'raw_data_dri_2','raw_data_fan_2','raw_data_dri_fan_2','ML_CNN_1_2',...
    'ML_CNN_2_2','MSIF_DARL_2','MSIF_ML_CNN_2'};
cell_file_name_3_dimension={'raw_data_dri_3','raw_data_fan_3','raw_data_dri_fan_3','ML_CNN_1_3',...
    'ML_CNN_2_3','MSIF_DARL_3','MSIF_ML_CNN_3'};
cell_file_list={cell_file_name_2_dimension,cell_file_name_3_dimension};

for i = 1:7
    file_name_2=cell_file_list{1}{i}
    matrix_2=cell_matrix_list{1}{i}
    plot_function(matrix_2, label, 2, file_name_2)
end

for i = 1:7
    file_name_3=cell_file_list{2}{i};
    matrix_3=cell_matrix_list{2}{i};
    plot_function(matrix_3, label, 3, file_name_3)
end

% plot_function(fc12, label, 2, 'fc1_final_three_2_dim')

