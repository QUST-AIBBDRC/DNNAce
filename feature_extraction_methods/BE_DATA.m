clear all
clc
input=importdata('E_coil.txt');
data=input(2:2:end,:);
[m,n]=size(data);
label1=ones(m/2,1);
label2=zeros(m/2,1);
label=[label1;label2];
out=[];
input=data;
for i=1:m
    protein=input{i};
    output =BE_feature(protein);
    out=[out;output];
    ouput=[];
end
save BE_E_coil.mat out