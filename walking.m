clc;
clear all;
close all;
accel = importdata('TYPE_ACCELEROMETER1487967188846.csv');
gyro = importdata('TYPE_GYROSCOPE1487967188929.csv');
magn = importdata('TYPE_MAGNETIC_FIELD1487967188894.csv');
l_accel = importdata('TYPE_LINEAR_ACCELERATION1487967188960.csv');
pres = importdata('TYPE_PRESSURE1487967188913.csv');
gps = importdata('TYPE_GPS1487892930346.csv');
grav = importdata('TYPE_GRAVITY1487967188977.csv');
orient = importdata('TYPE_ORIENTATION1487967188863.csv');
rot_vec = importdata('TYPE_ROTATION_VECTOR1487967188944.csv');
ini_time = gps.data(1,3);

%%
gps.data(:,3) = gps.data(:,3) - ini_time;
accel.data(:,4) = accel.data(:,4) - ini_time;
gyro.data(:,4) = gyro.data(:,4) - ini_time;
magn.data(:,4) = magn.data(:,4) - ini_time;
l_accel.data(:,4) = l_accel.data(:,4) - ini_time;
pres.data(:,2) = pres.data(:,2) - ini_time;
grav.data(:,4) = grav.data(:,4) - ini_time;
orient.data(:,4) = orient.data(:,4) - ini_time;
rot_vec.data(:,6) = rot_vec.data(:,6) - ini_time;

accel_data = accel.data(accel.data(:,4)>0,:);
gyro_data = gyro.data(gyro.data(:,4)>0,:);
magn_data = magn.data(magn.data(:,4)>0,:);
laccel_data = l_accel.data(l_accel.data(:,4)>0,:);
pres_data = pres.data(pres.data(:,2)>0,:);
grav_data = grav.data(grav.data(:,4)>0,:);
orient_data = orient.data(orient.data(:,4)>0,:);
rot_data = rot_vec.data(rot_vec.data(:,6)>0,:);

%%
t= gps.data(:,3);
tq = 0:10:max(t);
 vq = interp1(t,gps.data,tq);
 l= length(vq);
 i=1;
 j=1;
 k=1;
 n=1;
 for n=1:l
     if(accel_data(i,4)>= vq(n,3)-2 & (accel_data(i,4)< vq(n,3)+10))
         vq(n,4:7) = accel_data(i,:);
         while(accel_data(i,4)< vq(n,3)+10)
         i=i+1;
         end
     end
     
     if(gyro_data(j,4)>= vq(n,3)-2 & (gyro_data(j,4)< vq(n,3)+10))
         vq(n,8:11) = gyro_data(j,:);
         while(gyro_data(j,4)< vq(n,3)+10)
         j=j+1;
         end
     end
     
     if(magn_data(k,4)>= vq(n,3)-2 & (magn_data(k,4)< vq(n,3)+10))
         vq(n,12:15) = magn_data(k,:);
         while(magn_data(k,4)< vq(n,3)+10)
         k=k+1;
         end
     end
 end
 
 %%
 
 vq = vq(vq(:,15)>0,:);
 vq = vq(vq(:,7)>0,:);
 vq = vq(vq(:,11)>0,:);
 
 %%
 
 full = [vq(:,7) vq(:,4:6) vq(:,8:10) vq(:,12:14) vq(:,1:2)];
 %csvwrite('full_table.csv',full);
%  dlmwrite('myFile.csv',full,'precision','%.15f');
%m_per_deg_lat = 111132.954 - 559.822 * cos( 2 * latMid ) + 1.175 * cos( 4 * latMid);
% max = 34.06927887 min =34.06743383
%  m_per_deg_lat = 111132.954 - 559.822 * cos( 2 * latMid ) + 1.175 * cos( 4 * latMid);
% m_per_deg_lon = 111132.954 * cos ( latMid );

% double latMid, m_per_deg_lat, m_per_deg_lon, deltaLat, deltaLon,dist_m;
% 
% latMid = (Lat1+Lat2 )/2.0;  // or just use Lat1 for slightly less accurate estimate
% 
% m_per_deg_lat = 111132.954 - 559.822 * cos( 2.0 * latMid ) + 1.175 * cos( 4.0 * latMid);
% m_per_deg_lon = (3.14159265359/180 ) * 6367449 * cos ( latMid );
% 
% deltaLat = fabs(Lat1 - Lat2);
% deltaLon = fabs(Lon1 - Lon2);
% 
% dist_m = sqrt (  pow( deltaLat * m_per_deg_lat,2) + pow( deltaLon * m_per_deg_lon , 2) );