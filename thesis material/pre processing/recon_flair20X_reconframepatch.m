
%%%%% reconstruction with reconframe, 20X flair


clear all;
%%

setenv('TOOLBOX_PATH', '/bart/bin')
addpath(getenv('/bart/dir'));
addpath(('/dir/spm12'));

maindir='/dir/to/main/';
odir='/data/to/output/';
cd(maindir);

r=MRecon('tof.lab');
r.Parameter.Parameter2Read.typ = 1;
r.Parameter.Parameter2Read.Update;
r.ReadData;
% r.Parameter.Encoding.XRes=r.Parameter.Encoding.XReconRes;
    
    r.RandomPhaseCorrection;
    r.RemoveOversampling;
    r.PDACorrection;
    r.DcOffsetCorrection;
    r.MeasPhaseCorrection;
    r.SortData;
    r.RingingFilter;
  
r.ZeroFill;    
%%
    
s=MRsense('senseref.raw',r);
s.OutputSizeReformated=size(r.Data);
s.OutputSizeSensitivity=s.OutputSizeReformated;

s.Mask = 1;
s.Smooth = 1;
s.Extrapolate = 1;
s.Perform;
r.Parameter.Recon.Sensitivities = s;
r.Parameter.Recon.SENSERegStrength=0;
% r.Perform;
% calculated sensemaps are stored in: s.Sensitivity

%%
sref_nonnorm=bart('fft 7',s.ReformatedCoilData);
% ecalib=bart('ecalib -r 16 -k 4 -S -I',sref_nonnorm);
dims = size(r.Data);
slabs = dims(end);
smap_slabs = [];
for i=1:slabs
    smap_nonnorm=bart('caldir 50',sref_nonnorm(:,:,:,:,1,1,1,i));
  % smap_nonnorm_alt=bart('caldir 24', sref_nonnorm);
    smap_nonnorm=bart('fftshift 6',smap_nonnorm);
    smap_slabs=cat(5,smap_slabs,smap_nonnorm);
end   
   
% tmptmp=bart('fft -i 1',r.Data);
% tmptmp=squeeze(tmptmp(:,:,:,:,1,1,1,:));
% test5=sum(bart('fft -i 6',tmptmp).*(conj(smap_slabs)),4);
% test5=bart('fftshift 6',test5);

 kspace=squeeze(r.Data(:,:,:,:,1,1,1,:));
 if ~isfolder(fullfile(odir,'4RIM')); mkdir(fullfile(odir,'4RIM')); end
 for i=1:slabs
    dat=kspace(:,:,:,:,i);
    smap=smap_slabs(:,:,:,:,i);
    writecfl(fullfile(odir,strcat('RIM/_data_',int2str(i))),dat);
    writecfl(fullfile(odir,strcat('RIM/_smap_',int2str(i))),smap); 
 end
     
% function writecfl(filenameBase,data)
% writecfl(filenameBase, data)
%    Writes recon data to filenameBase.cfl (complex float)
%    and write the dimensions to filenameBase.hdr.
% 
%    Written to edit data for the Berkeley recon.
% 
% 2012 Joseph Y Cheng (jycheng@mrsrl.stanford.edu).
% 
%     dims = size(data);
%     writeReconHeader(filenameBase,dims);
% 
%     filename = strcat(filenameBase,'.cfl');
%     fid = fopen(filename,'w');
%     
%     if numel(dims)~=4
%         error('not supported')
%     end
%     d=dims(1:end-1);
%     for ii=1:dims(end)
%         data_o = zeros(prod(d)*2,1,'single');
%         data_o(1:2:end) = real(data(:,:,:,ii));
%         data_o(2:2:end) = imag(data(:,:,:,ii));
%         fwrite(fid,data_o,'float32');
%     end
%     
%     fclose(fid);
% end
% 
% function writeReconHeader(filenameBase,dims)
%     filename = strcat(filenameBase,'.hdr');
%     fid = fopen(filename,'w');
%     fprintf(fid,'# Dimensions\n');
%     for N=1:length(dims)
%         fprintf(fid,'%d ',dims(N));
%     end
%     if length(dims) < 5
%         for N=1:(5-length(dims))
%             fprintf(fid,'1 ');
%         end
%     end
%     fprintf(fid,'\n');
%     
%     fclose(fid);
% end
