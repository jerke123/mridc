clear; 

disp('setting paths')

setenv('TOOLBOX_PATH', '/bart/bin')
addpath(getenv('/dir/bart/matlab'));
addpath(('/dir/spm12'));

rawdir='/raw/dir/';
cd(rawdir);
parentproc='/parent/dir/';

doPreproc= 1; % should preproc be performed again? time-consuming

odir = '/output/dir/';
h5dir = '/h5_output/dir/';

if ~isfolder(odir); mkdir(odir); end
disp(['subject folder: ',rawdir]);

toflab=fullfile([rawdir,'tof.raw']);
ref=fullfile([rawdir,'senseref.raw']);

rfile=fullfile(odir,'senserefall_real.nii');
ifile=fullfile(odir,'senserefall_imag.nii');
rrfile=fullfile(odir,'rsenserefall_real.nii');
rifile=fullfile(odir,'rsenserefall_imag.nii');
 
if doPreproc==1
    disp('loading flair list-data and applying preprocessing steps');
    r = MRecon(toflab);
    r.Parameter.Parameter2Read.typ = 1;
    r.Parameter.Parameter2Read.Update;
    r.ReadData;
    
    r.RandomPhaseCorrection;
    r.RemoveOversampling;
    r.PDACorrection;
    r.DcOffsetCorrection;
    r.MeasPhaseCorrection;
    r.SortData;
    r.RingingFilter;
    
    r.Parameter.Encoding.XReconRes=r.Parameter.Encoding.XRes;
    r.Parameter.Encoding.YReconRes=r.Parameter.Encoding.YRes; 
    r.Parameter.Encoding.ZReconRes=r.Parameter.Encoding.ZRes;
    
    r.ZeroFill;
    
    dims = size(r.Data);
    slabs = dims(end);
    slices = dims(3);
    
    sx=size(r.Data,1); sy=size(r.Data,2); sz=size(r.Data,3); nrcoils=size(r.Data,4);
    rrx=r.Parameter.Encoding.XReconRes;
    rry=r.Parameter.Encoding.YReconRes; 
    rrz=r.Parameter.Encoding.ZReconRes;
    oversample_z = r.Parameter.Encoding.KzOversampling;
    rrz_os=round(rrz*oversample_z);  
    vol3=zeros(rrx, rry, rrz_os, nrcoils, slabs); % try padded
    vol3( rrx/2 - sx/2 +1:rrx/2 +sx/2 , rry/2 - sy/2 +1:rry/2 +sy/2, round(rrz_os/2 - sz/2) +1:round(rrz_os/2 +sz/2), :, :) = r.Data;
    tmptmp=bart('fft -i 1',vol3);
    writecfl(fullfile(odir,'4RIM/data'),tmptmp);
end
  %% start on senseref.cpx
if doPreproc==1
    disp('loading senseref.cpx and applying transforms');
    C=MRecon(ref);
    C2=MRsense('..._senserefscanV4.raw',r);
    C2.OutputSizeReformated=size(r.Data);
    C2.OutputSizeSensitivity=C2.OutputSizeReformated;
    C2.Mask = 1;
    C2.Smooth = 1;
    C2.Extrapolate = 1;
    C2.Perform;
    vol=C2.CoilData;
    
    tmp=flip(vol,2);
    tmp=permute(tmp,[2 1 3 4 5 6 7 8]); % from Z X Y C --> X Z Y C
    tmp=flip(flip(flip(tmp,1),2),3);
    tmp=squeeze(tmp(:,:,:,:,1,1,1,1));

    %%%% write senseref as nifti
    disp('bringing senseref to nifti for checks');
    sz=size(tmp);
    dim=sz(1:3); %currently AP-FH-LR axes

    sref_off = C.Parameter.Scan.Offcentre(1,:);
    voxelref= C.Parameter.Scan.RecVoxelSize;

    % NII-matlab convention mismatch: LR - PA - FH
    % currently: data in tmp are stored in
    % NB: we flipped the RL-dimension but not the FH direction, so flip the FH
    % offset
    offset = -[dim(3)*voxelref(3)/2+sref_off(2) dim(1)*voxelref(1)/2+sref_off(1)  dim(2)*voxelref(2)/2-sref_off(3)]; % sagittal
    
    % offcentre: half the number of voxels * voxel spacing;
    % TODO: spacing when even number of voxels
    a = [offset 0 pi/2 -pi/2 voxelref(1) voxelref(2) voxelref(3) 0 0 0];
    A = spm_matrix(a);
end

%% prepare/save data
if doPreproc==1
    
    sref_off = r.Parameter.Scan.Offcentre(1,:);
    voxelref= r.Parameter.Scan.RecVoxelSize;
    resolution = [r.Parameter.Encoding.YReconRes r.Parameter.Encoding.XReconRes r.Parameter.Encoding.ZReconRes];

    angle = r.Parameter.Scan.Angulation*pi/180;
    angle = [angle(1) angle(2) angle(3)+0.2];
    
    offset = -[resolution(3)*voxelref(3)/2+sref_off(2) resolution(1)*voxelref(1)/2+sref_off(1)  resolution(2)*voxelref(2)/2-sref_off(3)]; % sagittal
    offset = [offset(1)-20 offset(2) offset(3)+voxelref(3)*(185-slices/2.3)];
    
    
    
    %% create empty nifti to coregister flair
    for i=1:slabs       
        % save real and imaginary components separately in 4D
        n=nifti('/load/a/nifti.nii');% load unrelated file as a baseline nifti structure
        n.dat.fname=rfile;
        n.dat.scl_slope=max(abs(tmp(:)))/1e4;
        n.mat=A;
        n.mat0=n.mat;
        n.dat.dim=size(tmp);
        n.dat(:,:,:,:)=real(tmp);
        create(n);

        n.dat.fname=ifile;
        n.dat(:,:,:,:)=imag(tmp);
        create(n);
        
        offset = [offset(1) offset(2) offset(3)+voxelref(3)*(slices/2.3)];

        P = [offset angle voxelref 0 0 0];
        msin = spm_matrix(P);

        n.dat.fname = fullfile(odir,'dummy_tof.nii');
        n.dat.scl_slope = max(abs(tmp(:)))/1e4;
        n.mat = msin;
        n.dat.dim = [resolution(1) resolution(2) rrz_os size(tmp,4)];
        n.mat0=n.mat;
        n.dat(:,:,:,:)=zeros([resolution(1) resolution(2) rrz_os size(tmp,4)]);
        create(n);
        
    %% reslice to "flair"
        disp('reslicing senseref to tof');
        flags.mean=0;
        flags.which=1;
        spm_reslice({fullfile(odir,'dummy_tof.nii'),rfile,ifile},flags)
        
     %% bring sense back to raw data convention
        disp('load resliced nifti to continue processing')
        n=nifti(rrfile);
        rout=n.dat(:,:,:,:);
        n=nifti(rifile);
        iout=n.dat(:,:,:,:);

        s_resliced=rout+1i*iout; % make complex-valued output
        s_resliced2=permute(s_resliced,[2 1 3 4]);
        s_resliced3=flip(flip(s_resliced2,1),3);
        
     %% calculate sensemaps
        sref_nonnorm=bart('fft 7',s_resliced3);
        smap_nonnorm=bart('caldir 50',sref_nonnorm);
        smap_nonnorm_shifted=bart('fftshift 2',smap_nonnorm);
        
        smap_slabs=cat(5,smap_slabs,smap_nonnorm_shifted);
    end
    
 %% save sensemaps
    if ~isfolder(fullfile(odir,'4RIM')); mkdir(fullfile(odir,'4RIM')); end
    writecfl(fullfile(odir,'4RIM/sensemaps'),smap_slabs); 
end

    
%% reconstructions
% test3=sum(bart('fft -i 6',tmptmp).*(conj(smap_nonnorm_shifted)),4);
% test3=bart('fftshift 2',test3);

tmptmp=readcfl(fullfile(odir,'dir/data'));
sensemaps_pad_shift=readcfl(fullfile(odir,'dir/sensemaps'));

pics_3d = zeros(size(tmptmp,[1 2 3 5]));

for j=1:slabs
    for i=1:size(pics_3d,1)
        % -S (smoothing
        % -I (normalization)
        % -r 0.005
        pics_3d(i,:,:,j)=bart('pics -G 3 -l1 -r0.05',tmptmp(i,:,:,:,j),sensemaps_pad_shift(i,:,:,:,j));
    end
end
saveh5(pics_3d,fullfile(h5dir,'pics_recon.h5'),'RootName','pics')  

% clear tmptmp sensemaps_pad_shift





%% helper functions

% checkerboard, taken from script by LM Gottwald
function ch=create_checkerboard(s)
%s: size of checkerboard
% starts with -1 on top left corner
if length(s)==2
    ch=(((-1).^[1:s(1)]).*1i).'*(((-1).^[1:s(2)]).*1i);
elseif length(s)==3
    ch=(((-1).^[1:s(1)]).*1i).'*(((-1).^[1:s(2)]).*1i);
    ch=repmat(ch,[1 1 s(3)]);
    ch1d=(((-1).^[1:s(3)]).*1i).*(ones(1,s(3)).*-1i);
    ch1d=permute(ch1d,[1 3 2]);
    ch=bsxfun(@times,ch,ch1d);
elseif length(s)==1
    ch=(((-1).^[1:s(1)]).*1i).*(ones(1,s(1)).*-1i);
else
    error('unsupported size')
end

end

% 
function [matrix_perm,resolution]=matrix_from_sin(C2)

matrix_loca=zeros(3,3);

% [~,offcentr_incrs]=unix(['cat ',flairsin,' | grep ''loc_ap_rl_fh_offcentr_incrs'' | awk ''{print$6,$7,$8}'' '])

matrix_loca(3,:)= C2.RefScan.Parameter.Scan.Offcentre(1,:) ;

[~,row1]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$6}'' ']);
[~,row2]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$7}'' ']);
[~,row3]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_row_image_oris'' | awk ''{print$8}'' ']);
row1=splitlines(row1);
row2=splitlines(row2);
row3=splitlines(row3);
row1=row1(1);
row2=row2(1);
row3=row3(1);
row1=str2double(row1);
row2=str2double(row2);
row3=str2double(row3);
matrix_loca(1,:)= [row1 row2 row3] ; clear row1 row2 row3

[~,col1]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$6}'' ']);
[~,col2]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$7}'' ']);
[~,col3]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_col_image_oris'' | awk ''{print$8}'' ']);
col1=splitlines(col1);
col2=splitlines(col2);
col3=splitlines(col3);
col1=col1(1);
col2=col2(1);
col3=col3(1);
col1=str2double(col1);
col2=str2double(col2);
col3=str2double(col3);
matrix_loca(2,:)= [col1 col2 col3] ; clear col1 col2 col3

% [~,vox1]=unix(['cat ',tofsin,' | grep ''voxel_sizes'' | awk ''{print$6}'' ']);
% RecVoxelSize or AcqVoxelSize
voxel_sizes= C2.RefScan.Parameter.Scan.RecVoxelSize ;

% [~,res1]=unix(['cat ',tofsin,' | grep ''output_resolutions'' | awk ''{print$6}'' ']);
res1= C2.RefScan.Parameter.Encoding.XReconRes;
res2= C2.RefScan.Parameter.Encoding.YReconRes;
res3= C2.RefScan.Parameter.Encoding.ZReconRes;
resolution= [res1 res2 res3] ; clear res1 res2 res3

% [~,cc1]=unix(['cat ',tofsin,' | grep ''loc_ap_rl_fh_offcentres'' | awk ''{print$6}'' ']);
centre_coords= C2.RefScan.Parameter.Scan.Offcentre(1,:) ;

matrix_loca=matrix_loca';

matrix_loca(1,3)=-matrix_loca(1,3);            
matrix_loca(2,3)=-matrix_loca(2,3);
matrix_loca(3,1)=-matrix_loca(3,1);
matrix_loca(3,2)=-matrix_loca(3,2);

matrix(:,1)= matrix_loca(:,1)*voxel_sizes(1);
matrix(:,2)= matrix_loca(:,2)*voxel_sizes(2);
matrix(:,3)= matrix_loca(:,3); 

offset1 =  -resolution(1)/2*matrix(1,1) + ...
            -resolution(2)/2*matrix(1,2) + ...
            -resolution(3)/2*matrix(1,3) + ...
            +centre_coords(1);
offset2 =   -resolution(1)/2*matrix(2,1) + ...
            -resolution(2)/2*matrix(2,2) + ...
            -resolution(3)/2*matrix(2,3) + ...
            +centre_coords(2);
offset3 =   -resolution(1)/2*matrix(3,1) + ...
            -resolution(2)/2*matrix(3,2) + ...
            -resolution(3)/2*matrix(3,3) + ...
            +centre_coords(3);
        
matrix(:,4)= [offset1; offset2; offset3];

matrix_perm = zeros(size(matrix));
 matrix_perm(:,1)= matrix(:,3);
 matrix_perm(:,2)= matrix(:,1);
 matrix_perm(:,3)= matrix(:,2);
 matrix_perm(:,4)= matrix(:,4);
 
 matrix_perm = matrix_perm([2 1 3],:);
 matrix_perm(4,:)= [0 0 0 1];
end
