clear all
clc
input=importdata('E_coil.txt');
data=input(2:2:end,:);
[m,n]=size(data);
vector=[];
for i=1:m;
 vector= [vector;EBGW_yu(data{i})];
end
save EBGW_E_coil.mat vector


