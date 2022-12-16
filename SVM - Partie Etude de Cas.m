% Séparateurs à Vaste Marge pour la Classification
% Cas linéaire, non linéaire, marges "souples"

clear all
close all
global ikernel  % type de noyau choisi.     1: produit scalaire dans Rp (linéaire)
                                                         % 2: noyau gaussien
                                                            % Vous pouvez
                                                            % en ajouter
                                                            % d'autres

 load 'mensur' X lab     % chargement des données 
 data = readtable('mensur_eleves.csv');
 ikernel=3;             % choix du type de noyau
 marge_souple= true;  % choix marge souple ou pas
 C_souple= 100;          % coef souplesse de la marge souple


na=length(X); % nombre de points (d'apprentissage)



% gradient à pas constant pour problème dual
% On vous laisse quelques valeurs pour les paramettres reglables... à vous de voir
alph0=0.5*ones(na,1); % point de départ (0, ou autre): réglable
pasgrad=5e-5;         % pas du gradient : parametre réglable
u=ones(na,1);         % vecteur de 1
crit_arret=1;         % initialisation critere d'arret
npas=0;               % comptage du nombre de pas
npas_max=1000;      % garde-fou convergence : on arrete si le nombre d'iterations est trop grand
epsi=1e-5;            % seuil convergence
M = Compute_M(X,lab); % On assemble la matrice M



% boucle de gradient projeté
% cas où on a une marge rigide
if marge_souple == false
    alph = alph0;
    while npas <= npas_max || crit_arret > epsi
        alph0 = alph;
        grad = -M*alph0 + u;
        alph = alph0 + pasgrad*grad;
        alph = alph - (dot(lab,alph))*lab/(lab'*lab);
        alph = max(alph,0);
        crit_arret = norm(alph-alph0);
        nbit(npas +1) = npas;
        ecart(npas +1) = crit_arret;
        npas = npas +1;
    end
% cas où on a une marge souple
else
    alph = alph0;
    while npas <= npas_max || crit_arret > epsi
        alph0 = alph;
        grad = -M*alph0 + u;
        alph = alph0 + pasgrad*grad;
        alph = alph - (dot(lab,alph))*lab/(lab'*lab);
        alph = max(alph,0);
        alph = min(alph,C_souple);
        crit_arret = norm(alph-alph0);
        nbit(npas +1) = npas;
        ecart(npas +1) = crit_arret;
        npas = npas +1;
    end
end

% recherche des points supports
epsi=1e-5;
Points_support=find(alph>epsi)'; % voir help find

if marge_souple == true
    Points_support = find(alph>epsi & abs(C_souple - alph)>epsi) ;
end    

moy_b = zeros(length(Points_support), 1) ;
for i=1:length(Points_support)
    moy_b(i) = 1/lab(Points_support(i)) - prodw(X,lab,alph,X(:,Points_support(i))) ;
end


b=mean(moy_b);   %  on moyenne les b des points supports
%grid


Pseudo = [] ;
Results = [] ;

%X_el = table2array(data(:,2:3))' ; % Si on se limite aux deux premières variables%
X_el = table2array(data(:,2:6))'; % Si on utilise toutes les données
for k=1:15
    Results(1,k) =k;
    Results(2,k) = new_point(X_el(:,k),X,lab,alph,b) ;
end

disp(Results);
function NP = new_point(x,X,lab,alph,b)
if prodw(X,lab,alph,x) + b > 0
    NP = 1 ;
end
if prodw(X,lab,alph,x) + b <= 0
    NP = 0 ;
end
end


