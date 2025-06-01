%% SPC
% Reference: https://arxiv.org/abs/2505.11524

%% Generating training and testing dataset
clear all;close all
% Generate the input-output data by simulating the linear system
% Prediction horizon: N
% Measurement horizon: M
randn('state',1);
rand('state',1);

NT=1000;n=4;m=1; p=1; T = 0.1; Ntr=500;N=30;H=20;M=400;    
Ad=[0.5 0 0.05 0.1;0 0.7 0.6 0.4; 0.1 0.2 0.5 0.1;0.2 0.1 -0.1 0.1];Bd=[0.5;0.2;0.1;0.7];Cd=[1 0 0 0];Dd=0;                                
x=zeros(n,NT+1); x0=[0;0;0;0];     x(:,1)=x0;
u=kron(0.5*randn(1,NT/10),ones(1,10)); 
for k=1:NT                
    x(:,k+1)=Ad*x(:,k)+Bd*u(:,k);            % State equation
    y(:,k)=Cd*x(:,k)+Dd*u(:,k);              % Output equation
end           
UD=u(:,1:500);                               % Training input
YD=y(:,1:500)+0.05*randn(1,500);             % Training output           
Us=u(:,501:1000);                            % Testing input
Ys=y(:,501:1000)+0.05*randn(1,500);          % Testing output

%% Subspace based identification
Up=f_Hankel(UD,H,M,m);
Uf=f_Hankel(UD(:,H+1:end),N,M,m);
Yp=f_Hankel(YD,H,M,p);
Yf=f_Hankel(YD(:,H+1:end),N,M,p);


S=[Up;Yp;Uf];
P=Yf*pinv(S);
P1=P(:,1:H*m);
P2=P(:,H*m+1:H*m+H*p);
BY=P(:,H*m+H*p+1:end);
LW=[P1 P2];
[Uw,Sw,Vw]=svd(LW);

%% SPC 
NS=200; 
for k=1:NS+N+1
if k>=1 && k<=50
    Yr(k,1)=1;
elseif k>50 && k<=100    
    Yr(k,1)=0.7;
 elseif k>100 && k<=150    
    Yr(k)=0.5;
else    
    Yr(k,1)=1;
end
end
Qy=5*eye(p);QN=Qy;
Ru=0.1*eye(m);
QY=Qy;RU=Ru;
for i=1:N-1
  QY=blkdiag(QY,Qy); RU=blkdiag(RU,Ru);
end

Y=zeros(p,NS+N+1); Y(:,1:H)=YD(:,1:H);
Yk=zeros(p*(N+1)); 
U=zeros(m,NS+N); U(:,1:H)=UD(:,1:H);
umin=-5;umax=5;
Uk0=randn(m*N,1);
Uk1=ones(m*N,1);

% simulating system with SPC 
for k=H+1:NS
   Yrk=Yr(k+1:k+N,1); 
   Ypk=reshape(Y(:,k-H+1:k),[],1);
   Upk=reshape(U(:,k-H+1:k),[],1);
   fun3 = @(Uk)(Yrk-(P1*Upk+P2*Ypk+BY*Uk))'*QY*(Yrk-(P1*Upk+P2*Ypk+BY*Uk))+(Uk-Uk1)'*RU*(Uk-Uk1);
   F=[];g=[];Feq=[];geq=[];lb=[umin*ones(m*N,1)];ub=[umax*ones(m*N,1)];nonlcon=[];
   Uk=fmincon(fun3,Uk0,F,g,Feq,geq,lb,ub,nonlcon);
   U(k)=Uk(1:m,1);
   Yfk=P1*Upk+P2*Ypk+BY*Uk;
   Y(:,k+1)=Yfk(1:p,1);
   Uk0=Uk;
   Uk1=Uk;
end   

%% Plotting results
figure(1)
subplot(2,1,1)
time = (0:Ntr);
stairs(time(1:end-1),UD,'r.-','LineWidth',2) 
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$u_{k}$','Interpreter','latex','FontSize',18);
grid on
subplot(2,1,2)
plot(time(1:end-1),YD,'r.-','LineWidth',2) 
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$y_{k}$','Interpreter','latex','FontSize',18);
grid on
figure(2)
plot(Yr,'g', LineWidth=5)
hold on
plot(Y,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$y_{r_k}, \hat{y}_{k|k}$','Interpreter','latex','fontsize',18);
legend('$y_{r_k}$', '${y}_{k|k}$','Interpreter','latex','fontsize',14)
grid on
figure(3)
stairs(U,'r', LineWidth=2)
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$u_{k|k}$','Interpreter','latex','fontsize',18);
grid on


%% Function computing block Hankel matrix 
function H_V=f_Hankel(V,N,M,q)    
H_V=zeros(N*q,M);
for i=1:N
    for j=1:M
    H_V((i-1)*q+1:i*q,j)=V(:,i+j-1);
    end
end
end