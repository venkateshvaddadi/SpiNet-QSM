
clear all; 
close all; 
clc;
%%
% loading the input files

input_dir='data/dataset_1/input/'
output_dir='data/dataset_1/output/'

%%
input_phs=load(strcat(input_dir,'phs1.mat')).phs;
input_msk=load(strcat(input_dir,'msk1.mat')).msk;
cosmos_ground_truth=load(strcat(input_dir,'cos1.mat')).cos;
addpath('metrics/')

%%
% loading the output files

spinet_qsm_output=load(strcat(output_dir,'SpiNet_QSM_output.mat')).susc;
spinet_qsm_output = squeeze(spinet_qsm_output);

%%
ssim_measured= round(compute_ssim(spinet_qsm_output,cosmos_ground_truth), 4);      
rmse_measured = round(compute_rmse(spinet_qsm_output,cosmos_ground_truth), 4);      
psnr_measured= round(compute_psnr(spinet_qsm_output,cosmos_ground_truth), 4);      
hfen_measured= round(compute_hfen(spinet_qsm_output,cosmos_ground_truth), 4);   


N = size(input_phs);
spatial_res = [1 1 1];
[ky,kx,kz] = meshgrid(-N(1)/2:N(1)/2-1, -N(2)/2:N(2)/2-1, -N(3)/2:N(3)/2-1);
kx = (kx / max(abs(kx(:)))) / spatial_res(1);
ky = (ky / max(abs(ky(:)))) / spatial_res(2);
kz = (kz / max(abs(kz(:)))) / spatial_res(3);

% Compute magnitude of kernel and perform fftshift
k2 = kx.^2 + ky.^2 + kz.^2;
kernel = 1/3 - (kz.^2 ./ (k2 + eps)); % Z is the B0-direction
kernel = fftshift(kernel);

phi_x=real(ifftn(fftn(spinet_qsm_output).*kernel)).*single(input_msk);
diff=phi_x-input_phs;

model_loss=norm(diff(:));

%%
% display final results

disp('Comparison with Ground-Truth (COSMOS):')
fprintf('SSIM: %.4f \n',ssim_measured)
fprintf('pSNR: %.4f \n',psnr_measured)
fprintf('RMSE: %.4f \n',rmse_measured)
fprintf('HEFN: %.4f \n',hfen_measured)
fprintf('Model Loss: %.4f \n',model_loss)

%%
% making the output figure
% The figure will be saved at: data/output/
disp_fig(spinet_qsm_output, cosmos_ground_truth,output_dir);
