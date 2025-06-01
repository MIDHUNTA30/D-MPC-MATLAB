%% PEM based data-driven LMPC
% Reference: https://arxiv.org/abs/2505.11524

%% Generating training and testing dataset
clear all;close all
global UD n m p n NT D
randn('state',1);
rand('state',1);
% Generate the input-output data by simulating the linear system
NT=1000;n=4;m=1; p=1; T = 0.1; D=500;    
Ad=[0.5 0 0.05 0.1;0 0.7 0 0.04; 0 0 0.55 0.1;0.2 0.1 0 0.1];Bd=[0.5;0;0.1;0.7];Cd=[1 0 0 0];Dd=0;                                
x=zeros(n,NT+1); x0=[0;0;0;0];     x(:,1)=x0;
u=kron(0.5*randn(1,NT/20),ones(1,20)); 
for k=1:NT                
    x(:,k+1)=Ad*x(:,k)+Bd*u(:,k);            % State equation
    y(:,k)=Cd*x(:,k)+Dd*u(:,k);              % Output equation
end           
UD=u(:,1:500);                               % Training input
YD=y(:,1:500)+0.01*randn(1,500);             % Training output                     
Us=u(:,501:1000);                            % Testing input
Ys=y(:,501:1000)+0.01*randn(1,500);          % Testing output

%% State space model identification using PEM
theta_x=0.1*rand(n+p,n+m+1);  % theta_x contains the hyperparameters in the model
for i=1:2
  fun = @(theta_x)trace((YD-(theta_x(n+1:n+p,1:n)*fxp(theta_x,UD,D)))'*(YD-(theta_x(n+1:n+p,1:n)*fxp(theta_x,UD,D))));
  options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
  [theta_x,fval,flag]=fminunc(fun,theta_x,options);       
end
A=theta_x(1:n,1:n); B=theta_x(1:n,n+1:n+m); C=theta_x(n+1:n+p,1:n);

% Prediction on training data  
Xp=zeros(n,D+1);  Xp(:,1)=theta_x(1:n,n+m+1); Yp=zeros(p,D); 
for k=1:D
Xp(:,k+1)=A*Xp(:,k)+B*UD(:,k);
Yp(:,k)=C*Xp(:,k);
end  

%Estimating the initial state
Xts10=randn(n,1);
options1 = optimoptions('fsolve','MaxIterations',1e7,'MaxFunctionEvaluations',1e6)
fun0=@(Xts1)[Ys(:,1)-(C*Xts1); Ys(:,2)-(C*A*Xts1+C*B*Us(:,1))];
[Xts1,fval0,flag0] = fsolve(fun0,Xts10,options1);

% Prediction on testing data    -
Xsp=zeros(n,500);     Xsp(:,1)=Xts1; 
for k=1:500
Xsp(:,k+1)=A*Xsp(:,k)+B*Us(:,k);
Ysp(:,k)=C*Xsp(:,k);
end 

%% MPC using PEM based model
N=10; NS=200; 
for k=1:NS
if k>=1 && k<=50
    Yr(k)=1;
elseif k>50 && k<=100    
    Yr(k)=0.7;
 elseif k>100 && k<=150    
    Yr(k)=0.5;
else    
    Yr(k)=1;
end
      
Xr0=randn(n+m,1);
fun1=@(Xr1)[Yr(k)-C*Xr1(1:n,1);Xr1(1:n,1)-A*Xr1(1:n,1)+B*Xr1(n+1:n+m,1)];
[Xr1,fval1,flag1] = fsolve(fun1,Xr0,options1);
Xr(:,k)=Xr1(1:n,1);
Ur(:,k)=Xr1(n+1:n+m,1);
end

Qx=2*eye(n);
Qu=3*eye(m);
xk=zeros(n,1);
Y=zeros(p,NS+1);
x0=xk;
X=zeros(n,NS+1); X(:,1)=x0;
Xk=zeros(n,N+1); Xk(:,1)=x0;
U=zeros(m,NS); %u(:,1)=Ur(:,1);
Y=zeros(p,NS);
umin=-5;umax=5;
Uk0=-1*rand(m,N);
Uk1=zeros(m,N);
% Simulating system with PEM based LMPC 
for k=1:NS
   theta_x(1:n,n+m+1)=X(:,k);  
   Xrk=Xr(:,k);Urk=Ur(:,k); Yrk= Yr(:,k);
   fun3 = @(Uk)trace((Xrk-fxp(theta_x,Uk,N))'*Qx*(Xrk-fxp(theta_x,Uk,N)))+trace((Uk-Uk1)'*Qu*(Uk-Uk1));
   F=[];g=[];Feq=[];geq=[];lb=[umin*ones(m,N)];ub=[umax*ones(m,N)];nonlcon=[];
   Uk=fmincon(fun3,Uk0,F,g,Feq,geq,lb,ub,nonlcon);
   U(k)=Uk(:,1);
   X(:,k+1)=A*X(:,k)+B*U(:,k);
   Y(:,k)=C*X(:,k);
   Uk0=Uk;
   Uk1=Uk;
end   

%% Plotting results
figure(1)
subplot(2,1,1)
plot(UD,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$u_{k}$','Interpreter','latex','fontsize',18);
grid on
subplot(2,1,2)
plot(YD,'r', LineWidth=3)
hold on
plot(Yp,'g', LineWidth=2)
xlabel('$k$','Interpreter','latex','fontsize',18);ylabel('$y_{k}, \hat{y}_{k}$','Interpreter','latex','fontsize',18);
grid on
figure(2)
plot(Yr,'g', LineWidth=5)
hold on
plot(Y,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$y_{r_k}, {y}_{k|k}$','Interpreter','latex','fontsize',18);
legend('$y_{r_k}$', '${y}_{k|k}$','Interpreter','latex','fontsize',14)
grid on
figure(3)
stairs(U,'r', LineWidth=2)
xlabel('$k$','Interpreter','latex','FontSize',18);ylabel('$u_{k|k}$','Interpreter','latex','fontsize',18);
grid on

%% Predicting the states using fxp 
function Xhat=fxp(theta_x,u,Np)    
global NT N n m Ts n p D
A=theta_x(1:n,1:n); B=theta_x(1:n,n+1:n+m); C=theta_x(n+1:n+p,1:n);
Xhat=zeros(n,Np); Xhat(:,1)=theta_x(1:n,n+m+1);
for i=1:Np-1
    Xhat(:,i+1)=A*Xhat(:,i)+B*u(:,i);
end
end
