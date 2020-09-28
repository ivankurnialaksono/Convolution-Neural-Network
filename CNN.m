% catatan
% Data harus csv dg kolom terakir berupa kelas (1 dan 0), urut 1 dulubaru 0

clear all
clc
%Load
d              = csvread('C:\Users\Ivan\Desktop\durut.csv');

%Normalisasi data EEG
dataEEGnormalized  = zscore(d(:,1:(end-3)));
t                  = d(:,(end-2));

for i = 1:length(t(:,1));
    tt = t(i,1);
    if tt==1
        kelas(i,:) = [1 0];
    else
        kelas(i,:) = [0 1];
    end
end

data = [dataEEGnormalized(:,:) t kelas];
Ld  = length(data(1,:));
Nd  = length(data(:,1));
N1  = 480; %input user
N0  = 480; %input user

data1             = data(1:N1,:);
data0             = data(N1+1:Nd,:);

%ratio split
Nt1               = round(0.8*(length(data1(:,1))));
Nt0               = round(0.8*(length(data0(:,1))));
Ntrain            = Nt1+Nt0;

mix               = [data1(1:Nt1,:); data0(1:Nt0,:)];
tTrain            = mix(1:Ntrain ,end-2);
kTrain            = mix(1:Ntrain ,(end-1):end);
%Zero Padding data EEG
input             = [zeros(Ntrain,4) mix(:,1:(end-3)) zeros(Ntrain,4)];

%kernel gaussian
K5 = [0.06136 0.24477 0.38774 0.24477 0.06136];
K3 = [0.27901 0.44198 0.27901];
tic
%Conv1 (1 x 5) stride 2
n1 = (round((Ld+2)/2))-1;
for n=1:n1;
    for m=1:Ntrain;
    c1(m,n) = dot(input(m,2*n-1:2*n+3),K5);
    end
end
%ReLu
for n = 1:n1;
    for m = 1:Ntrain
    r1(m,n) = (max((c1(m,n)),0)) ;
    end
end
% max pooling 1x3 stride 2
n2=round((n1-1)/2)-1;
for n=1:n2;
    for m=1:Ntrain;
    p1(m,n) = max(r1(m,2*n-1:2*n+1));
    end
end
%Conv3 1x3 stride 1
p2padd = [zeros(Ntrain,2) p1 zeros(Ntrain,2)];
n3 = n2+2;
for n=1:n3;
    for m=1:Ntrain;
    c2(m,n) = dot(p2padd(m,n:n+2),K3);
    end
end
%ReLu
for n = 1:n3;
    for m = 1:Ntrain;
    r2(m,n) = (max((c2(m,n)),0)) ;
    end
end
%maxpool 1x3 stride 2
n4 = round(n3-1)/2;
for nfinal=1:n4;
    for m=1:Ntrain;
    p2(m,nfinal) = max(r2(m,2*nfinal-1:2*nfinal+1));
    end
end

NFINAL = nfinal;
%FULLY CONECTED LAYER

%input for lullly connected layer
x = p1(:,:);
% Initialize the bias
biasv = ones(1,30);
biasw = [1 1]; 
% Learning coefficient
coeff = 0.4;
% Number of learning iterations
iterations = 200;
% Calculate random weights
rand('state',sum(100*clock));
v1 = -1 +2.*rand(1,nfinal);
v2 = -1 +2.*rand(1,nfinal);
v3 = -1 +2.*rand(1,nfinal);
v4 = -1 +2.*rand(1,nfinal);
v5 = -1 +2.*rand(1,nfinal);
v6 = -1 +2.*rand(1,nfinal);
v7 = -1 +2.*rand(1,nfinal);
v8 = -1 +2.*rand(1,nfinal);
v9 = -1 +2.*rand(1,nfinal);
v10 = -1 +2.*rand(1,nfinal);
v11 = -1 +2.*rand(1,nfinal);
v12 = -1 +2.*rand(1,nfinal);
v13 = -1 +2.*rand(1,nfinal);
v14 = -1 +2.*rand(1,nfinal);
v15 = -1 +2.*rand(1,nfinal);
v16 = -1 +2.*rand(1,nfinal);
v17 = -1 +2.*rand(1,nfinal);
v18 = -1 +2.*rand(1,nfinal);
v19 = -1 +2.*rand(1,nfinal);
v20 = -1 +2.*rand(1,nfinal);
v21 = -1 +2.*rand(1,nfinal);
v22 = -1 +2.*rand(1,nfinal);
v23 = -1 +2.*rand(1,nfinal);
v24 = -1 +2.*rand(1,nfinal);
v25 = -1 +2.*rand(1,nfinal);
v26 = -1 +2.*rand(1,nfinal);
v27 = -1 +2.*rand(1,nfinal);
v28 = -1 +2.*rand(1,nfinal);
v29 = -1 +2.*rand(1,nfinal);
v30 = -1 +2.*rand(1,nfinal);
w1 = -1 +2.*rand(1,30);
w2 = -1 +2.*rand(1,30);

for i = 1:iterations;
out   = zeros(219,1);
numIn = length (x(:,1));
for j = 1:numIn
% Hidden layer
z1net = sum(x(j,1:nfinal).*v1)+biasv(1,1);
z2net = sum(x(j,1:nfinal).*v2)+biasv(1,2);
z3net = sum(x(j,1:nfinal).*v3)+biasv(1,3);
z4net = sum(x(j,1:nfinal).*v4)+biasv(1,4);
z5net = sum(x(j,1:nfinal).*v5)+biasv(1,5);
z6net = sum(x(j,1:nfinal).*v6)+biasv(1,6);
z7net = sum(x(j,1:nfinal).*v7)+biasv(1,7);
z8net = sum(x(j,1:nfinal).*v8)+biasv(1,8);
z9net = sum(x(j,1:nfinal).*v9)+biasv(1,9);
z10net = sum(x(j,1:nfinal).*v10)+biasv(1,10);
z11net = sum(x(j,1:nfinal).*v11)+biasv(1,11);
z12net = sum(x(j,1:nfinal).*v12)+biasv(1,12);
z13net = sum(x(j,1:nfinal).*v13)+biasv(1,13);
z14net = sum(x(j,1:nfinal).*v14)+biasv(1,14);
z15net = sum(x(j,1:nfinal).*v15)+biasv(1,15);
z16net = sum(x(j,1:nfinal).*v16)+biasv(1,16);
z17net = sum(x(j,1:nfinal).*v17)+biasv(1,17);
z18net = sum(x(j,1:nfinal).*v18)+biasv(1,18);
z19net = sum(x(j,1:nfinal).*v19)+biasv(1,19);
z20net = sum(x(j,1:nfinal).*v20)+biasv(1,20);
z21net = sum(x(j,1:nfinal).*v21)+biasv(1,21);
z22net = sum(x(j,1:nfinal).*v22)+biasv(1,22);
z23net = sum(x(j,1:nfinal).*v23)+biasv(1,23);
z24net = sum(x(j,1:nfinal).*v24)+biasv(1,24);
z25net = sum(x(j,1:nfinal).*v25)+biasv(1,25);
z26net = sum(x(j,1:nfinal).*v26)+biasv(1,26);
z27net = sum(x(j,1:nfinal).*v27)+biasv(1,27);
z28net = sum(x(j,1:nfinal).*v28)+biasv(1,28);
z29net = sum(x(j,1:nfinal).*v29)+biasv(1,29);
z30net = sum(x(j,1:nfinal).*v30)+biasv(1,30);

% Send data through sigmoid function 1/1+e^-x
Z1 = 1/(1+exp(-z1net));
Z2 = 1/(1+exp(-z2net));
Z3 = 1/(1+exp(-z3net));
Z4 = 1/(1+exp(-z4net));
Z5 = 1/(1+exp(-z5net));
Z6 = 1/(1+exp(-z6net));
Z7 = 1/(1+exp(-z7net));
Z8 = 1/(1+exp(-z8net));
Z9 = 1/(1+exp(-z9net));
Z10 = 1/(1+exp(-z10net));
Z11 = 1/(1+exp(-z11net));
Z12 = 1/(1+exp(-z12net));
Z13 = 1/(1+exp(-z13net));
Z14 = 1/(1+exp(-z14net));
Z15 = 1/(1+exp(-z15net));
Z16 = 1/(1+exp(-z16net));
Z17 = 1/(1+exp(-z17net));
Z18 = 1/(1+exp(-z18net));
Z19 = 1/(1+exp(-z19net));
Z20 = 1/(1+exp(-z20net));
Z21 = 1/(1+exp(-z21net));
Z22 = 1/(1+exp(-z22net));
Z23 = 1/(1+exp(-z23net));
Z24 = 1/(1+exp(-z24net));
Z25 = 1/(1+exp(-z25net));
Z26 = 1/(1+exp(-z26net));
Z27 = 1/(1+exp(-z27net));
Z28 = 1/(1+exp(-z28net));
Z29 = 1/(1+exp(-z29net));
Z30 = 1/(1+exp(-z30net));

z = [Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 Z14 Z15 Z16 Z17 Z18 Z19 Z20 Z21 Z22 Z23 Z24 Z25 Z26 Z27 Z28 Z29 Z30];

O1net = sum(z.*w1)+biasw(1,1);
O2net = sum(z.*w2)+biasw(1,2);

%output
O1 = 1/(1+exp(-O1net));
O2 = 1/(1+exp(-O2net));

O(j,:) =[O1 O2];
maxo = max(O1,O2);

if O1==maxo
    OO(j,:) = [1 0];
else if O2==maxo
    OO(j,:) = [0 1];
end
end

if OO(j,:) == [1 0];
    T(j) = 1;
else if OO(j,:) == [0 1];
        T(j) = 0;
    end
end
%delta weight (W)
%determine delta weight using gradient descent
dw11 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z1;
dw12 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z2;
dw13 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z3;
dw14 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z4;
dw15 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z5;
dw16 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z6;
dw17 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z7;
dw18 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z8;
dw19 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z9;
dw110 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z10;
dw111 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z11;
dw112 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z12;
dw113 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z13;
dw114 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z14;
dw115 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z15;
dw116 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z16;
dw117 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z17;
dw118 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z18;
dw119 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z19;
dw120 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z20;
dw121 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z21;
dw122 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z22;
dw123 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z23;
dw124 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z24;
dw125 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z25;
dw126 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z26;
dw127 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z27;
dw128 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z28;
dw129 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z29;
dw130 = coeff*(kelas(j,1)-O1)*O1*(1-O1)*Z30;

dw21 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z1;
dw22 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z2;
dw23 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z3;
dw24 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z4;
dw25 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z5;
dw26 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z6;
dw27 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z7;
dw28 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z8;
dw29 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z9;
dw210 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z10;
dw211 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z11;
dw212 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z12;
dw213 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z13;
dw214 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z14;
dw215 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z15;
dw216 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z16;
dw217 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z17;
dw218 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z18;
dw219 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z19;
dw220 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z20;
dw221 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z21;
dw222 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z22;
dw223 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z23;
dw224 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z24;
dw225 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z25;
dw226 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z26;
dw227 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z27;
dw228 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z28;
dw229 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z29;
dw230 = coeff*(kelas(j,2)-O2)*O2*(1-O2)*Z30;

%delta bias untuk W
%menggunakan metode gradient descent
dbiasw1 = coeff*(kelas(j,1)-O1)*O1*(1-O1);
dbiasw2 = coeff*(kelas(j,2)-O2)*O2*(1-O2);

%delta weigh (V)
%menggunakan metode gradient descent
for i=1: nfinal;
dv1(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z1*w1(1,1)*(1-Z1)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z1*w2(1,1)*(1-Z1)*x(j,i)));
dv2(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z2*w1(1,2)*(1-Z2)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z2*w2(1,2)*(1-Z2)*x(j,i)));
dv3(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z3*w1(1,3)*(1-Z3)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z3*w2(1,3)*(1-Z3)*x(j,i)));
dv4(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z4*w1(1,4)*(1-Z4)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z4*w2(1,4)*(1-Z4)*x(j,i)));
dv5(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z5*w1(1,5)*(1-Z5)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z5*w2(1,5)*(1-Z5)*x(j,i)));
dv6(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z6*w1(1,6)*(1-Z6)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z6*w2(1,6)*(1-Z6)*x(j,i)));
dv7(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z7*w1(1,7)*(1-Z7)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z7*w2(1,7)*(1-Z7)*x(j,i)));
dv8(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z8*w1(1,8)*(1-Z8)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z8*w2(1,8)*(1-Z8)*x(j,i)));
dv9(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z9*w1(1,9)*(1-Z9)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z9*w2(1,9)*(1-Z9)*x(j,i)));
dv10(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z10*w1(1,10)*(1-Z10)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z10*w2(1,10)*(1-Z10)*x(j,i)));
dv11(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z11*w1(1,11)*(1-Z11)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z11*w2(1,11)*(1-Z11)*x(j,i)));
dv12(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z12*w1(1,12)*(1-Z12)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z12*w2(1,12)*(1-Z12)*x(j,i)));
dv13(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z13*w1(1,13)*(1-Z13)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z13*w2(1,13)*(1-Z13)*x(j,i)));
dv14(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z14*w1(1,14)*(1-Z14)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z14*w2(1,14)*(1-Z14)*x(j,i)));
dv15(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z15*w1(1,15)*(1-Z15)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z15*w2(1,15)*(1-Z15)*x(j,i)));
dv16(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z16*w1(1,16)*(1-Z16)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z16*w2(1,16)*(1-Z16)*x(j,i)));
dv17(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z17*w1(1,17)*(1-Z17)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z17*w2(1,17)*(1-Z17)*x(j,i)));
dv18(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z18*w1(1,18)*(1-Z18)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z18*w2(1,18)*(1-Z18)*x(j,i)));
dv19(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z19*w1(1,19)*(1-Z19)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z19*w2(1,19)*(1-Z19)*x(j,i)));
dv20(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z20*w1(1,20)*(1-Z20)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z20*w2(1,20)*(1-Z20)*x(j,i)));
dv21(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z21*w1(1,21)*(1-Z21)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z21*w2(1,21)*(1-Z21)*x(j,i)));
dv22(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z22*w1(1,22)*(1-Z22)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z22*w2(1,22)*(1-Z22)*x(j,i)));
dv23(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z23*w1(1,23)*(1-Z23)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z23*w2(1,23)*(1-Z23)*x(j,i)));
dv24(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z24*w1(1,24)*(1-Z24)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z24*w2(1,24)*(1-Z24)*x(j,i)));
dv25(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z25*w1(1,25)*(1-Z25)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z25*w2(1,25)*(1-Z25)*x(j,i)));
dv26(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z26*w1(1,26)*(1-Z26)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z26*w2(1,26)*(1-Z26)*x(j,i)));
dv27(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z27*w1(1,27)*(1-Z27)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z27*w2(1,27)*(1-Z27)*x(j,i)));
dv28(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z28*w1(1,28)*(1-Z28)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z28*w2(1,28)*(1-Z28)*x(j,i)));
dv29(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z29*w1(1,29)*(1-Z29)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z29*w2(1,29)*(1-Z29)*x(j,i)));
dv30(j,i) = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z30*w1(1,30)*(1-Z30)*x(j,i))+((kelas(j,2)-O2)*O2*(1-O2)*Z30*w2(1,30)*(1-Z30)*x(j,i)));

end
%delta bias untuk V
%menggunakan metode gradient descent
dbiasv1 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z1*w1(1,1)*(1-Z1))+((kelas(j,2)-O2)*O2*(1-O2)*Z1*w2(1,1)*(1-Z1)));
dbiasv2 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z2*w1(1,2)*(1-Z2))+((kelas(j,2)-O2)*O2*(1-O2)*Z2*w2(1,2)*(1-Z2)));
dbiasv3 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z3*w1(1,3)*(1-Z3))+((kelas(j,2)-O2)*O2*(1-O2)*Z3*w2(1,3)*(1-Z3)));
dbiasv4 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z4*w1(1,4)*(1-Z4))+((kelas(j,2)-O2)*O2*(1-O2)*Z4*w2(1,4)*(1-Z4)));
dbiasv5 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z5*w1(1,5)*(1-Z5))+((kelas(j,2)-O2)*O2*(1-O2)*Z5*w2(1,5)*(1-Z5)));
dbiasv6 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z6*w1(1,6)*(1-Z6))+((kelas(j,2)-O2)*O2*(1-O2)*Z6*w2(1,6)*(1-Z6)));
dbiasv7 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z7*w1(1,7)*(1-Z7))+((kelas(j,2)-O2)*O2*(1-O2)*Z7*w2(1,7)*(1-Z7)));
dbiasv8 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z8*w1(1,8)*(1-Z8))+((kelas(j,2)-O2)*O2*(1-O2)*Z8*w2(1,8)*(1-Z8)));
dbiasv9 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z9*w1(1,9)*(1-Z9))+((kelas(j,2)-O2)*O2*(1-O2)*Z9*w2(1,9)*(1-Z9)));
dbiasv10 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z10*w1(1,10)*(1-Z10))+((kelas(j,2)-O2)*O2*(1-O2)*Z10*w2(1,10)*(1-Z10)));
dbiasv11 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z11*w1(1,11)*(1-Z11))+((kelas(j,2)-O2)*O2*(1-O2)*Z11*w2(1,11)*(1-Z11)));
dbiasv12 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z12*w1(1,12)*(1-Z12))+((kelas(j,2)-O2)*O2*(1-O2)*Z12*w2(1,12)*(1-Z12)));
dbiasv13 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z13*w1(1,13)*(1-Z13))+((kelas(j,2)-O2)*O2*(1-O2)*Z13*w2(1,13)*(1-Z13)));
dbiasv14 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z14*w1(1,14)*(1-Z14))+((kelas(j,2)-O2)*O2*(1-O2)*Z14*w2(1,14)*(1-Z14)));
dbiasv15 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z15*w1(1,15)*(1-Z15))+((kelas(j,2)-O2)*O2*(1-O2)*Z15*w2(1,15)*(1-Z15)));
dbiasv16 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z16*w1(1,16)*(1-Z16))+((kelas(j,2)-O2)*O2*(1-O2)*Z16*w2(1,16)*(1-Z16)));
dbiasv17 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z17*w1(1,17)*(1-Z17))+((kelas(j,2)-O2)*O2*(1-O2)*Z17*w2(1,17)*(1-Z17)));
dbiasv18 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z18*w1(1,18)*(1-Z18))+((kelas(j,2)-O2)*O2*(1-O2)*Z18*w2(1,18)*(1-Z18)));
dbiasv19 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z19*w1(1,19)*(1-Z19))+((kelas(j,2)-O2)*O2*(1-O2)*Z19*w2(1,19)*(1-Z19)));
dbiasv20 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z20*w1(1,20)*(1-Z20))+((kelas(j,2)-O2)*O2*(1-O2)*Z20*w2(1,20)*(1-Z20)));
dbiasv21 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z21*w1(1,21)*(1-Z21))+((kelas(j,2)-O2)*O2*(1-O2)*Z21*w2(1,21)*(1-Z21)));
dbiasv22 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z22*w1(1,22)*(1-Z22))+((kelas(j,2)-O2)*O2*(1-O2)*Z22*w2(1,22)*(1-Z22)));
dbiasv23 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z23*w1(1,23)*(1-Z23))+((kelas(j,2)-O2)*O2*(1-O2)*Z23*w2(1,23)*(1-Z23)));
dbiasv24 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z24*w1(1,24)*(1-Z24))+((kelas(j,2)-O2)*O2*(1-O2)*Z24*w2(1,24)*(1-Z24)));
dbiasv25 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z25*w1(1,25)*(1-Z25))+((kelas(j,2)-O2)*O2*(1-O2)*Z25*w2(1,25)*(1-Z25)));
dbiasv26 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z26*w1(1,26)*(1-Z26))+((kelas(j,2)-O2)*O2*(1-O2)*Z26*w2(1,26)*(1-Z26)));
dbiasv27 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z27*w1(1,27)*(1-Z27))+((kelas(j,2)-O2)*O2*(1-O2)*Z27*w2(1,27)*(1-Z27)));
dbiasv28 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z28*w1(1,28)*(1-Z28))+((kelas(j,2)-O2)*O2*(1-O2)*Z28*w2(1,28)*(1-Z28)));
dbiasv29 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z29*w1(1,29)*(1-Z29))+((kelas(j,2)-O2)*O2*(1-O2)*Z29*w2(1,29)*(1-Z29)));
dbiasv30 = coeff*(((kelas(j,1)-O1)*O1*(1-O1)*Z30*w1(1,30)*(1-Z30))+((kelas(j,2)-O2)*O2*(1-O2)*Z30*w2(1,30)*(1-Z30)));

% Add weight changes to original weights
% And use the new weights to repeat process.
% Update bobot dan bias

w1(1,1) = w1(1,1) + dw11;
w1(1,2) = w1(1,2) + dw12;
w1(1,3) = w1(1,3) + dw13;
w1(1,4) = w1(1,4) + dw14;
w1(1,5) = w1(1,5) + dw15;
w1(1,6) = w1(1,6) + dw16;
w1(1,7) = w1(1,7) + dw17;
w1(1,8) = w1(1,8) + dw18;
w1(1,9) = w1(1,9) + dw19;
w1(1,10) = w1(1,10) + dw110;
w1(1,11) = w1(1,11) + dw111;
w1(1,12) = w1(1,12) + dw112;
w1(1,13) = w1(1,13) + dw113;
w1(1,14) = w1(1,14) + dw114;
w1(1,15) = w1(1,15) + dw115;
w1(1,16) = w1(1,16) + dw116;
w1(1,17) = w1(1,17) + dw117;
w1(1,18) = w1(1,18) + dw118;
w1(1,19) = w1(1,19) + dw119;
w1(1,20) = w1(1,20) + dw120;
w1(1,21) = w1(1,21) + dw121;
w1(1,22) = w1(1,22) + dw122;
w1(1,23) = w1(1,23) + dw123;
w1(1,24) = w1(1,24) + dw124;
w1(1,25) = w1(1,25) + dw125;
w1(1,26) = w1(1,26) + dw126;
w1(1,27) = w1(1,27) + dw127;
w1(1,28) = w1(1,28) + dw128;
w1(1,29) = w1(1,29) + dw129;
w1(1,30) = w1(1,30) + dw130;

w2(1,1) = w2(1,1) + dw21;
w2(1,2) = w2(1,2) + dw22;
w2(1,3) = w2(1,3) + dw23;
w2(1,4) = w2(1,4) + dw24;
w2(1,5) = w2(1,5) + dw25;
w2(1,6) = w2(1,6) + dw26;
w2(1,7) = w2(1,7) + dw27;
w2(1,8) = w2(1,8) + dw28;
w2(1,9) = w2(1,9) + dw29;
w2(1,10) = w2(1,10) + dw210;
w2(1,11) = w2(1,11) + dw211;
w2(1,12) = w2(1,12) + dw212;
w2(1,13) = w2(1,13) + dw213;
w2(1,14) = w2(1,14) + dw214;
w2(1,15) = w2(1,15) + dw215;
w2(1,16) = w2(1,16) + dw216;
w2(1,17) = w2(1,17) + dw217;
w2(1,18) = w2(1,18) + dw218;
w2(1,19) = w2(1,19) + dw219;
w2(1,20) = w2(1,20) + dw220;
w2(1,21) = w2(1,21) + dw221;
w2(1,22) = w2(1,22) + dw222;
w2(1,23) = w2(1,23) + dw223;
w2(1,24) = w2(1,24) + dw224;
w2(1,25) = w2(1,25) + dw225;
w2(1,26) = w2(1,26) + dw226;
w2(1,27) = w2(1,27) + dw227;
w2(1,28) = w2(1,28) + dw228;
w2(1,29) = w2(1,29) + dw229;
w2(1,30) = w2(1,30) + dw230;

biasw(1,1) = biasw(1,1) + dbiasw1;
biasw(1,2) = biasw(1,2) + dbiasw2;

W1Save = w1(:,:);
W2Save = w2(:,:);
BiasWsave = biasw(:,:);

for i = 1:nfinal;
v1(1,i) = v1(1,i) + dv1(j,i);
v2(1,i) = v2(1,i) + dv2(j,i);
v3(1,i) = v3(1,i) + dv3(j,i);
v4(1,i) = v4(1,i) + dv4(j,i);
v5(1,i) = v5(1,i) + dv5(j,i);
v6(1,i) = v6(1,i) + dv6(j,i);
v7(1,i) = v7(1,i) + dv7(j,i);
v8(1,i) = v8(1,i) + dv8(j,i);
v9(1,i) = v9(1,i) + dv9(j,i);
v10(1,i) = v10(1,i) + dv10(j,i);
v11(1,i) = v11(1,i) + dv11(j,i);
v12(1,i) = v12(1,i) + dv12(j,i);
v13(1,i) = v13(1,i) + dv13(j,i);
v14(1,i) = v14(1,i) + dv14(j,i);
v15(1,i) = v15(1,i) + dv15(j,i);
v16(1,i) = v16(1,i) + dv16(j,i);
v17(1,i) = v17(1,i) + dv17(j,i);
v18(1,i) = v18(1,i) + dv18(j,i);
v19(1,i) = v19(1,i) + dv19(j,i);
v20(1,i) = v20(1,i) + dv20(j,i);
v21(1,i) = v21(1,i) + dv21(j,i);
v22(1,i) = v22(1,i) + dv22(j,i);
v23(1,i) = v23(1,i) + dv23(j,i);
v24(1,i) = v24(1,i) + dv24(j,i);
v25(1,i) = v25(1,i) + dv25(j,i);
v26(1,i) = v26(1,i) + dv26(j,i);
v27(1,i) = v27(1,i) + dv27(j,i);
v28(1,i) = v28(1,i) + dv28(j,i);
v29(1,i) = v29(1,i) + dv29(j,i);
v30(1,i) = v30(1,i) + dv30(j,i);
end

biasv(1,1) = biasv(1,1) + dbiasv1;
biasv(1,2) = biasv(1,2) + dbiasv2;
biasv(1,3) = biasv(1,3) + dbiasv3;
biasv(1,4) = biasv(1,4) + dbiasv4;
biasv(1,5) = biasv(1,5) + dbiasv5;
biasv(1,6) = biasv(1,6) + dbiasv6;
biasv(1,7) = biasv(1,7) + dbiasv7;
biasv(1,8) = biasv(1,8) + dbiasv8;
biasv(1,9) = biasv(1,9) + dbiasv9;
biasv(1,10) = biasv(1,10) + dbiasv10;
biasv(1,11) = biasv(1,11) + dbiasv11;
biasv(1,12) = biasv(1,12) + dbiasv12;
biasv(1,13) = biasv(1,13) + dbiasv13;
biasv(1,14) = biasv(1,14) + dbiasv14;
biasv(1,15) = biasv(1,15) + dbiasv15;
biasv(1,16) = biasv(1,16) + dbiasv16;
biasv(1,17) = biasv(1,17) + dbiasv17;
biasv(1,18) = biasv(1,18) + dbiasv18;
biasv(1,19) = biasv(1,19) + dbiasv19;
biasv(1,20) = biasv(1,20) + dbiasv20;
biasv(1,21) = biasv(1,21) + dbiasv21;
biasv(1,22) = biasv(1,22) + dbiasv22;
biasv(1,23) = biasv(1,23) + dbiasv23;
biasv(1,24) = biasv(1,24) + dbiasv24;
biasv(1,25) = biasv(1,25) + dbiasv25;
biasv(1,26) = biasv(1,26) + dbiasv26;
biasv(1,27) = biasv(1,27) + dbiasv27;
biasv(1,28) = biasv(1,28) + dbiasv28;
biasv(1,29) = biasv(1,29) + dbiasv29;
biasv(1,30) = biasv(1,30) + dbiasv30;

end

vSave = [v1(:,:);v2(:,:);v3(:,:);v4(:,:);v5(:,:);v6(:,:);v7(:,:);v8(:,:);v9(:,:);v10(:,:);v11(:,:);v12(:,:);v13(:,:);v14(:,:);v15(:,:);v16(:,:);v17(:,:);v18(:,:);v19(:,:);v20(:,:);v21(:,:);v22(:,:);v23(:,:);v24(:,:);v25(:,:);v26(:,:);v27(:,:);v28(:,:);v29(:,:);v30(:,:)];
biasvSave = biasv(:,:);

SALAH1 = sum(abs(tTrain'-T));
error = SALAH1/Ntrain;
%berhenti jika akurasi lebih dari 96
if error < 0.01
    break
end
end

Plus = tTrain'+T;
Minus = tTrain'-T;
Correct1 = sum(Plus==2);
Correct0 = sum(Plus==0);
Incorrect1 = sum(Minus==1);
Incorrect0 = sum(Minus==-1);

%confutions matrikx
akurasiTRAINING = ((Ntrain-SALAH1)/Ntrain)*100
ConfutionMatrixTraining =[Correct1 Incorrect1; Incorrect0 Correct0]
toc


%load data
Ntes1             = N1-Nt1;
Ntes0             = N0-Nt0;
Ntest             = Ntes1+Ntes0;
mix2              = [data1((Nt1+1):N1,:); data0((Nt0+1):N0,:)];
tTest             = mix2(1:Ntest ,end-2);

for i = 1:length(tTest (:,1));
    tt = tTest (i,1);
    if tt==1
        kTest(i,:) = [1 0];
    else
        kTest(i,:) = [0 1];
    end
end

inputTest         = [zeros(Ntest,4) mix2(:,1:(end-3)) zeros(Ntest,4)];

tic
%Conv1 (1 x 5) stride 2
n1 = (round((Ld+2)/2))-1;
for n=1:n1;
    for m=1:Ntest;
    C1(m,n) = dot(inputTest(m,2*n-1:2*n+3),K5);
    end
end

%ReLu
for n = 1:n1;
    for m = 1:Ntest;
    R1(m,n) = (max((C1(m,n)),0)) ;
    end
end

% max pooling 1x3 stride 2
n2=round((n1-1)/2)-1;
for n=1:n2;
    for m=1:Ntest;
    P1(m,n) = max(R1(m,2*n-1:2*n+1));
    end
end
%Conv2 1x3 stride 1
P2padd = [zeros(Ntest,2) P1 zeros(Ntest,2)];
n3 = n2+2;
for n=1:n3;
    for m=1:Ntest;
    C2(m,n) = dot(P2padd(m,n:n+2),K3);
    end
end
%ReLu
for n = 1:n3;
    for m = 1:Ntest;
    R2(m,n) = (max((C2(m,n)),0)) ;
    end
end
%maxpool 1x3 stride 2
n4 = round(n3-1)/2;
for nfinal=1:n4;
    for m=1:Ntest;
    P2(m,nfinal) = max(R2(m,2*nfinal-1:2*nfinal+1));
    end
end

%full connected
%input data fully connected layer 
biasv  = biasvSave;
biasw  = BiasWsave;
w1     = W1Save;
w2     = W2Save;
V      = vSave;

v1     = V(1,:);
v2     = V(2,:);
v3     = V(3,:);
v4     = V(4,:);
v5     = V(5,:);
v6     = V(6,:);
v7     = V(7,:);
v8     = V(8,:);
v9     = V(9,:);
v10     = V(10,:);
v11     = V(11,:);
v12     = V(12,:);
v13     = V(13,:);
v14     = V(14,:);
v15     = V(15,:);
v16     = V(16,:);
v17     = V(17,:);
v18     = V(18,:);
v19     = V(19,:);
v20     = V(20,:);
v21     = V(21,:);
v22     = V(22,:);
v23     = V(23,:);
v24     = V(24,:);
v25     = V(25,:);
v26     = V(26,:);
v27     = V(27,:);
v28     = V(28,:);
v29     = V(29,:);
v30     = V(30,:);

x2 = P2(:,:);
for i = 1;
numIn2 = length (x2(:,1));
for j = 1:numIn2
% Hidden layer

z1net = sum(x2(j,1:nfinal).*v1)+biasv(1,1);
z2net = sum(x2(j,1:nfinal).*v2)+biasv(1,2);
z3net = sum(x2(j,1:nfinal).*v3)+biasv(1,3);
z4net = sum(x2(j,1:nfinal).*v4)+biasv(1,4);
z5net = sum(x2(j,1:nfinal).*v5)+biasv(1,5);
z6net = sum(x2(j,1:nfinal).*v6)+biasv(1,6);
z7net = sum(x2(j,1:nfinal).*v7)+biasv(1,7);
z8net = sum(x2(j,1:nfinal).*v8)+biasv(1,8);
z9net = sum(x2(j,1:nfinal).*v9)+biasv(1,9);
z10net = sum(x2(j,1:nfinal).*v10)+biasv(1,10);
z11net = sum(x2(j,1:nfinal).*v11)+biasv(1,11);
z12net = sum(x2(j,1:nfinal).*v12)+biasv(1,12);
z13net = sum(x2(j,1:nfinal).*v13)+biasv(1,13);
z14net = sum(x2(j,1:nfinal).*v14)+biasv(1,14);
z15net = sum(x2(j,1:nfinal).*v15)+biasv(1,15);
z16net = sum(x2(j,1:nfinal).*v16)+biasv(1,16);
z17net = sum(x2(j,1:nfinal).*v17)+biasv(1,17);
z18net = sum(x2(j,1:nfinal).*v18)+biasv(1,18);
z19net = sum(x2(j,1:nfinal).*v19)+biasv(1,19);
z20net = sum(x2(j,1:nfinal).*v20)+biasv(1,20);
z21net = sum(x2(j,1:nfinal).*v21)+biasv(1,21);
z22net = sum(x2(j,1:nfinal).*v22)+biasv(1,22);
z23net = sum(x2(j,1:nfinal).*v23)+biasv(1,23);
z24net = sum(x2(j,1:nfinal).*v24)+biasv(1,24);
z25net = sum(x2(j,1:nfinal).*v25)+biasv(1,25);
z26net = sum(x2(j,1:nfinal).*v26)+biasv(1,26);
z27net = sum(x2(j,1:nfinal).*v27)+biasv(1,27);
z28net = sum(x2(j,1:nfinal).*v28)+biasv(1,28);
z29net = sum(x2(j,1:nfinal).*v29)+biasv(1,29);
z30net = sum(x2(j,1:nfinal).*v30)+biasv(1,30);

% Send data through sigmoid function 1/1+e^-x
% Note that sigma is a different m file
% that I created to run this operation
Z1 = 1/(1+exp(-z1net));
Z2 = 1/(1+exp(-z2net));
Z3 = 1/(1+exp(-z3net));
Z4 = 1/(1+exp(-z4net));
Z5 = 1/(1+exp(-z5net));
Z6 = 1/(1+exp(-z6net));
Z7 = 1/(1+exp(-z7net));
Z8 = 1/(1+exp(-z8net));
Z9 = 1/(1+exp(-z9net));
Z10 = 1/(1+exp(-z10net));
Z11 = 1/(1+exp(-z11net));
Z12 = 1/(1+exp(-z12net));
Z13 = 1/(1+exp(-z13net));
Z14 = 1/(1+exp(-z14net));
Z15 = 1/(1+exp(-z15net));
Z16 = 1/(1+exp(-z16net));
Z17 = 1/(1+exp(-z17net));
Z18 = 1/(1+exp(-z18net));
Z19 = 1/(1+exp(-z19net));
Z20 = 1/(1+exp(-z20net));
Z21 = 1/(1+exp(-z21net));
Z22 = 1/(1+exp(-z22net));
Z23 = 1/(1+exp(-z23net));
Z24 = 1/(1+exp(-z24net));
Z25 = 1/(1+exp(-z25net));
Z26 = 1/(1+exp(-z26net));
Z27 = 1/(1+exp(-z27net));
Z28 = 1/(1+exp(-z28net));
Z29 = 1/(1+exp(-z29net));
Z30 = 1/(1+exp(-z30net));

z = [Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Z13 Z14 Z15 Z16 Z17 Z18 Z19 Z20 Z21 Z22 Z23 Z24 Z25 Z26 Z27 Z28 Z29 Z30];

O1net = sum(z.*w1)+biasw(1,1);
O2net = sum(z.*w2)+biasw(1,2);


O1 = 1/(1+exp(-O1net));
O2 = 1/(1+exp(-O2net));

O(j,:) =[O1 O2];
maxo = max(O1,O2);

if O1==maxo
    OO(j,:) = [1 0];
else if O2==maxo
    OO(j,:) = [0 1];
end
end

if OO(j,:) == [1 0];
    T2(j) = 1;
else if OO(j,:) == [0 1];
        T2(j) = 0;
    end
end

end
end

plus = tTest'+T2;
minus = tTest'-T2;
correct1 = sum(plus==2);
correct0 = sum(plus==0);
incorrect1 = sum(minus==1);
incorrect0 = sum(minus==-1);

%resume = matriks hasil
PERCENTAGEsplit = '80%'
SALAH = sum(abs(tTest'-T2));
AKURASI = ((Ntest-SALAH)/Ntest)*100
ConfutionMatrix =[correct1 incorrect1; incorrect0 correct0]
toc







