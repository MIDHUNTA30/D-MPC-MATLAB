%% RNN based data-driven NMPC
% Implemented on continuous stirred tank reactor (CSTR) system
% Reference: https://arxiv.org/abs/2505.11524

%% Generating training and testing dataset
clear all;close all
global UD n m p h 
randn('state',1);
rand('state',1);
NT=1000;n=2;m=1; p=1; D=500; Ds=500;
B = 22;Da = 0.082;beta = 3.0; T=0.1;
x=zeros(n,NT+1); x0=[0;0];     x(:,1)=x0;
u=kron(-0.5*rand(1,NT/50),ones(1,50)); 
% The CSTR state equation is used for generating the training and testing data
 for k=1:NT        
    x(:,k+1)=x(:,k)+T*[-x(1,k)+Da*(1-x(1,k))*exp(x(2,k)); -x(2,k) + B*Da*(1-x(1,k))*exp(x(2,k))-beta*(x(2,k)-u(:,k))];  % state equaton is of the form x(k+1)=f(x(k),u(k)) 
    y(:,k)=x(2,k);              
 end                               

UD=u(:,1:500);                               % Training input
YD=y(:,1:500)+0.01*randn(1,500);             % Training output                     
Us=u(:,501:1000);                            % Testing input
Ys=y(:,501:1000)+0.01*randn(1,500);          % Testing output

%% RNN model training
n1=3;    % n1 is the number of neurons in the first hidden layer of RNN
theta_f=rand(n1,2*p+m);      % theta_f contains the hyperparameters in the model
y0=YD(:,1);
for i=1:2
  % Training the RNN model
  fun = @(theta_f)trace((YD-f_yp(theta_f,y0,UD,D))'*(YD-f_yp(theta_f,y0,UD,D)));    % Loss function for RNN model training
  options = optimoptions('fminunc','MaxIterations',1e6,'MaxFunctionEvaluations',1e6,'OptimalityTolerance',1e-5);
  [theta_f,fval,flag]=fminunc(fun,theta_f,options);  
end
 % Performance evaluation on training and testing data
% Predicting the training output with RNN    
Yp=zeros(p,D);Yp(:,1)=y0;   
for k=1:D-1
Yp(:,k+1)=theta_f(:,p+m+1:2*p+m)'*tanh(theta_f(:,1:p)*Yp(:,k)+theta_f(:,p+1:p+m)*UD(:,k));
end 
% Predicting the testing output with RNN   
Ysp=zeros(p,Ds); Ysp(:,1)=Ys(:,1); 
for k=1:Ds-1
Ysp(:,k+1)=theta_f(:,p+m+1:2*p+m)'*tanh(theta_f(:,1:p)*Ysp(:,k)+theta_f(:,p+1:p+m)*Us(:,k));
end 

MSEp=mse(YD-Yp);    % Training error
MSEs=mse(Ys-Ysp);   % Testing error

%% RNN-based NMPC
N=10; NS=200; 
% Defining the output reference
for k=1:NS
if k>=1 && k<=50
    Yr(k)=1;
elseif k>50 && k<=100    
    Yr(k)=0.5;
 elseif k>100 && k<=150    
    Yr(k)=0.7;
else    
    Yr(k)=1;
end
end
% Defining NMPC parameters
Qy=2*eye(p);
Qu=3*eye(m);
y0=0;
Y=zeros(p,NS+1); Y(:,1)=y0;      % Contains the output vectors y_k|k for MPC
Yk=zeros(p,N+1); Yk(:,1)=y0;
U=zeros(m,NS);                   % Contains the input vectors u_k|k for  MPC
umin=-2;umax=0;
Uk0=-1*rand(m,N);
Uk1=zeros(m,N);
% simulating system with RNN based NMPC 
for k=1:NS
   Yrk=Yr(:,k);
   fun3 = @(Uk)trace((Yrk-f_yp(theta_f,Y(:,k),Uk,N))'*Qy*(Yrk-f_yp(theta_f,Y(:,k),Uk,N)))+trace((Uk-Uk1)'*Qu*(Uk-Uk1));
   F=[];g=[];Feq=[];geq=[];lb=[umin*ones(m,N)];ub=[umax*ones(m,N)];nonlcon=[];
   Uk=fmincon(fun3,Uk0,F,g,Feq,geq,lb,ub,nonlcon);
   U(k)=Uk(:,1);
   Y(:,k+1)=theta_f(:,p+m+1:2*p+m)'*tanh(theta_f(:,1:p)*Y(:,k)+theta_f(:,p+1:p+m)*U(:,k));
   Uk0=Uk;
   Uk1=Uk;
end   

%% Plotting response
figure(1)
plot(YD,'r', LineWidth=3)
hold on
plot(Yp,'g', LineWidth=2)
xlabel('$k$','Interpreter','latex');ylabel('$y_{k}, \hat{y}_{k}$','Interpreter','latex');
grid on
figure(2)
plot(Yr,'g', LineWidth=5)
hold on
plot(Y,'r', LineWidth=3)
xlabel('$k$','Interpreter','latex');ylabel('$y_{r_k}, y_{k|k}$','Interpreter','latex');
legend('$y_{r_k}$', '$y_{k|k}$','Interpreter','latex')
grid on
figure(3)
plot(U,'r', LineWidth=2)
xlabel('$k$','Interpreter','latex');ylabel('$u_{k|k}$','Interpreter','latex');
grid on

%% Predicting the outputs using f_yp 
function Yhat=f_yp(theta_f,yk,u,NP)    
global n m Ts h p
Yhat=zeros(p,NP); Yhat(:,1)=yk;
for i=1:NP-1
    Yhat(:,i+1)=theta_f(:,p+m+1:2*p+m)'*tanh(theta_f(:,1:p)*Yhat(:,i)+theta_f(:,p+1:p+m)*u(:,i));
end
end
