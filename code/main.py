import numpy as np
from pr3_utils import *
import numpy as np
from scipy.linalg import expm
import time

# Load the measurements
filename = "../data/10.npz"
t,features,lin_vel, ang_vel,K,b,imu_T_cam = load_data(filename)
# flip signs of IMU y,z measurements
lin_vel[1:,:] = -lin_vel[1:,:]
ang_vel[1:,:] = -ang_vel[1:,:]
t=t[0]

def hat_map(vec):
	hatmap = np.zeros((3,3))
	hatmap[0, 1] = -vec[2]
	hatmap[0, 2] = vec[1]
	hatmap[1, 0] = vec[2]
	hatmap[1, 2] = -vec[0]
	hatmap[2, 0] = -vec[1]
	hatmap[2, 1] = vec[0]
	return hatmap

def hat_map_4(v,w):
	hatmap_4 = np.zeros((4, 4))
	hatmap_4[3, 3] = 0
	hatmap_4[0, 1] = -w[2]
	hatmap_4[0, 2] = w[1]
	hatmap_4[1, 0] = w[2]
	hatmap_4[1, 2] = -w[0]
	hatmap_4[2, 0] = -w[1]
	hatmap_4[2, 1] = w[0]
	hatmap_4[0:3, 3] = v
	return hatmap_4

def motion_model(T, tau, w, v):
	twist_hat = hat_map_4(v,w)
	return T @ expm(tau * twist_hat), twist_hat[0:3,0:3]

def twist_hat_covar_6(tau,w_hat,v):
	twist_hat_covar = np.zeros((6, 6))
	twist_vel = np.zeros((3, 3))
	twist_vel = hat_map(v)
	twist_hat_covar[0:3, 0:3] = twist_hat_covar[3:, 3:] = w_hat
	twist_hat_covar[0:3, 3:] = twist_vel
	return twist_hat_covar

def covar_prediction_step(covar,tau,w_hat,v):
	twist_hat_covar = twist_hat_covar_6(tau, w_hat, v)
	return expm(-tau*twist_hat_covar) @ covar @ expm((-tau*twist_hat_covar).T)

# global variables
STRIDE = 20
NOISE_SCALE = 0
list_remove_outliers = [1]
for remove_outliers in list_remove_outliers:
	# subset the observations at every time step to downsample the data
	features = features[:,::STRIDE]
	OUTLIER_DIST = 180
	OUTLIER_THRESH = np.linalg.norm(np.array([OUTLIER_DIST,OUTLIER_DIST]))
	num_outliers_init = 0
	n = t.shape[0]
	# initialize landmark means at 0
	# mu_map_all = np.zeros((n,4,features.shape[1]))
	mu_map = np.zeros((4,features.shape[1]))
	# initialize landmark covariances at identity
	covar_map = np.eye(3*features.shape[1])
	# covar_map_all = np.zeros((n,3*features.shape[1],3*features.shape[1]))
	# covar_map_all[0] = covar_map_init
	oRr = np.array([[0,-1,0],[0,0,-1],[1,0,0]])
	# invert optical to imu transformation matrix given in the project files
	oTi = np.linalg.inv(imu_T_cam)
	# construct Ks stereo camera calibration matrix
	Ks = np.zeros((4,4))
	Ks[0:2,0:3] = K[0:2,0:3]
	Ks[2:,0:3] = K[0:2,0:3]
	Ks[2,3] = -K[0,0]*b
	P = np.zeros((3,4))
	P[0:3,0:3] = np.eye(3)
	poses = np.zeros((n,4,4))
	slam_covar = np.eye(3*features.shape[1]+6)
	poses[0] = np.eye(4)
	set_seen_inds = set()
	start = time.time()
	for i in range(n-1):
	# for i in range(10):
		print(i)
		# (a) IMU Localization via EKF Prediction
		tau = t[i + 1] - t[i]
		updated_pose, w_hat = motion_model(poses[i], tau, ang_vel[:, i], lin_vel[:, i])
		poses[i+1] = updated_pose
		# covar_map = covar_prediction_step(covar_map, tau, w_hat, lin_vel[:, i])
		# add noise as a matrix where the variance of the noise for angular and linear velocity is scaled by tau
		# noise_ang_vel = np.random.normal(0,0.005,(3,))*0.01*ang_vel[:, i]
		# noise_lin_vel = np.random.normal(0,0.01,(3,))*0.01*lin_vel[:,i]
		# noise_6_hatmap = np.zeros((6,6))
		# noise_6_hatmap[:3,:3] = noise_6_hatmap[3:,3:] = hat_map(noise_ang_vel)
		# noise_6_hatmap[:3,3:] = hat_map(noise_lin_vel)
		lin_vel_noise = np.random.normal(0,NOISE_SCALE*np.abs(lin_vel[:,i]),(3,))
		w_noise = np.random.normal(0,NOISE_SCALE*np.abs(ang_vel[:,i]),(3,))
		# prediction step for covar updates the off diagonal map-robot correlations; treat each block separately
		# but note the map covariance doesn't change in the prediction step as map is static
		# robot covar
		# control input noise method
		slam_covar[3*features.shape[1]:, 3*features.shape[1]:] = covar_prediction_step(
			slam_covar[3*features.shape[1]:, 3*features.shape[1]:], tau, hat_map(ang_vel[:,i]+w_noise), lin_vel[:, i] + lin_vel_noise
		)
		# # hat map matrix addition noise method
		# # slam_covar[3*features.shape[1]:, 3*features.shape[1]:] = covar_prediction_step(
		# # 	slam_covar[3*features.shape[1]:, 3*features.shape[1]:], tau, w_hat, lin_vel[:, i]
		# # ) + noise_6_hatmap
		F = expm(-tau * twist_hat_covar_6(tau, w_hat, lin_vel[:, i]))
		# off-diagonal top right
		slam_covar[:3*features.shape[1],3*features.shape[1]:] = slam_covar[:3*features.shape[1],3*features.shape[1]:] @ F.T
		# off diagonal bottom left
		slam_covar[3*features.shape[1]:,:3*features.shape[1]] = F @ slam_covar[3*features.shape[1]:, :3 * features.shape[1]]

		inv_pose = np.zeros((4, 4))
		inv_pose[0:3, 0:3] = updated_pose[0:3, 0:3].T
		inv_pose[0:3, 3] = -updated_pose[0:3, 0:3].T @ updated_pose[0:3, 3]
		inv_pose[3, 3] = 1
		oTi_inv_pose = oTi @ inv_pose

		# (b) Landmark Mapping via EKF Update

		# first find out how many good observations there are for time t
		good_obs_inds = sorted(
			list(set(range(features.shape[1])) - set(np.unique(np.where((features[:, :, i].T == [-1, -1, -1, -1]))[0]))))
		good_obs_features = features[:, good_obs_inds, i]
		# now initialize landmarks that are being seen for the first time
		# if the feature has not been seen before, initialize it by solving the stereo camera model
		inds_to_pop = set()
		# print(good_obs_inds)
		for j,ind in enumerate(good_obs_inds):
			if ind not in set_seen_inds:
				# initialize the landmark at index j in good_obs_inds since it hasn't been seen before by solving stereo
				# camera model
				pixel_coords = features[:,ind,i]
				disp = pixel_coords[0] - pixel_coords[2]
				# linear system of equations solved by hand
				z = -Ks[2,3]/disp
				x = (pixel_coords[0]-Ks[0,2])*z/Ks[0,0]
				y = (pixel_coords[1]-Ks[1,2])*z/Ks[1,1]
				optical_landmark_coords_init = np.array([x,y,z,1])
				# world_landmark_coords_init = updated_pose @ imu_T_cam @ optical_landmark_coords_init
				world_landmark_coords_init = poses[i] @ imu_T_cam @ optical_landmark_coords_init
				# compute euclidean distance from robot to landmark
				robot_landmark_dist = np.linalg.norm(updated_pose[0:2,3] - world_landmark_coords_init[0:2])
				# also pop it from the good observation indices since we don't want to update the landmark positions
				# for those indices the first time we see it
				inds_to_pop.add(ind)
				# if the world coords of the landmark are too far from the current robot position, don't initialize it
				if remove_outliers:
					if robot_landmark_dist > OUTLIER_THRESH:
						num_outliers_init += 1
						continue
				# initialize the landmark position
				mu_map[:, ind] = world_landmark_coords_init
				# add this index to the set so that we update it next time see it rather than initialize it again
				set_seen_inds.add(ind)
			else:
				pass
		if remove_outliers:
			# deal with outliers that were initialized within the threshold but got pushed out due to update steps
			# simply check the positions of landmarks corresponding to good_obs_inds and compute distance to current robot pose
			dist_good_obs_curr_pos = np.linalg.norm(np.reshape(np.repeat(updated_pose[0:2,3],len(good_obs_inds)),(len(good_obs_inds),2),order='F')
													- mu_map[0:2,good_obs_inds].T, axis=1)
			arr_outlier_landmark_ix = np.where(dist_good_obs_curr_pos > OUTLIER_THRESH)[0]
			for m in list(arr_outlier_landmark_ix):
				# for each outlier, simply reposition the outlier so that it sits at a dist of THRESH away from the robot in arbitrary direction
				mu_map[0, m] = updated_pose[0, 3] + OUTLIER_DIST
				mu_map[1, m] = updated_pose[1, 3] + OUTLIER_DIST

		good_obs_inds = list(set(good_obs_inds) - inds_to_pop)
		# compute z_tilda
		N_t = len(good_obs_inds)
		# H should be initialized with zeros as 4N_t * 3M
		H = np.zeros((4 * N_t, 3 * features.shape[1]))
		# only compute for the landmarks that were observed in this time step
		if mu_map[:, good_obs_inds][0].shape[0] == 0:
			continue
		pi_map_arg = oTi_inv_pose @ mu_map[:,good_obs_inds]
		pi_map = pi_map_arg/pi_map_arg[2]
		z_til = Ks @ pi_map
		# compute H
		# create a loop and compute H for each landmark separately
		# for j in range(len(good_obs_inds)):
		# 	pi_map_arg = oTi_inv_pose @ mu_map[:,good_obs_inds[j]]
		# 	deriv_pi_map = np.zeros((4,4))
		# 	deriv_pi_map[0,0] = deriv_pi_map[1,1] = deriv_pi_map[3,3] = 1
		# 	deriv_pi_map[0,2] = -pi_map_arg[0]/pi_map_arg[2]
		# 	deriv_pi_map[1, 2] = -pi_map_arg[1] / pi_map_arg[2]
		# 	deriv_pi_map[3, 2] = -pi_map_arg[3] / pi_map_arg[2]
		# 	deriv_pi_map /= pi_map_arg[2]
		# 	H_landmark = Ks @ deriv_pi_map @ oTi_inv_pose @ P.T
		# 	H[4*j:4*j + 4, good_obs_inds[j]:good_obs_inds[j] + 3] = H_landmark
		#
		# # compute Kalman gain; shape = 3M x 4N_t
		# # define noise vector v using 0 mean, V covariance where V is Nt x Nt
		# V = np.diag(np.ones((N_t,)) * 5)
		# K_gain = covar_map @ H.T @ np.linalg.inv((H @ covar_map @ H.T) + np.kron(np.eye(4), V))
		#
		# # update map means - but first, flatten z_til to 4Nt x 1, features to 4Nt x 1 in column major order so that
		# # kalman gain multiplies along 4Nt dimension
		# mu_map[0:3] += np.reshape((K_gain @ (features[:,good_obs_inds,i] - z_til).flatten(order='F')),
		# 								  (3,features.shape[1]),order='F')
		# covar_map = (np.eye(3*features.shape[1],3*features.shape[1]) - (K_gain @ H)) @ covar_map

		# (c) Visual-Inertial SLAM

		H_slam = np.zeros((4*N_t,6+3*features.shape[1]))
		# block column matrix with blocks of 4x6 H_robot_i matrices
		H_robot = np.zeros((4*N_t,6))
		for j in range(len(good_obs_inds)):
			operator_arg = inv_pose @ mu_map[:, good_obs_inds[j]]
			operator = np.zeros((4,6))
			operator[0:3,0:3] = np.eye(3)
			operator[0:3,3:] = -hat_map(operator_arg)
			pi_map_arg = oTi @ operator_arg
			deriv_pi_map = np.zeros((4, 4))
			deriv_pi_map[0, 0] = deriv_pi_map[1, 1] = deriv_pi_map[3, 3] = 1
			deriv_pi_map[0, 2] = -pi_map_arg[0] / pi_map_arg[2]
			deriv_pi_map[1, 2] = -pi_map_arg[1] / pi_map_arg[2]
			deriv_pi_map[3, 2] = -pi_map_arg[3] / pi_map_arg[2]
			deriv_pi_map /= pi_map_arg[2]
			H_landmark = Ks @ deriv_pi_map @ oTi_inv_pose @ P.T
			H[4 * j:4 * j + 4, good_obs_inds[j]*3:good_obs_inds[j]*3 + 3] = H_landmark
			H_robot_j = -Ks @ deriv_pi_map @ oTi @ operator
			# add the individual H for each landmark into the block column vector for H robot
			H_robot[j*4: j*4 + 4, :] = H_robot_j
		# add in the H for robot into the combined slam H
		H_slam[:, 3 * features.shape[1]:] = H_robot
		# now insert the map H
		H_slam[:, :3 * features.shape[1]] = H
		V_slam = np.diag(np.ones((N_t,)) * 5)
		# calculate the Kalman gain using H_slam and covar_slam; shape = 3M+6 x 4N_t
		K_slam = slam_covar @ H_slam.T @ np.linalg.inv((H_slam @ slam_covar @ H_slam.T) + np.kron(np.eye(4), V_slam))

		# index the K_gain variable to update robot pose mean and visible landmark means separately
		# map landmark mean update
		mu_map[0:3] += np.reshape((K_slam[:3*features.shape[1],:] @ (features[:,good_obs_inds,i] - z_til).flatten(order='F')),
										  (3,features.shape[1]),order='F')
		# robot pose mean update - index K_slam to use only the robot portion (lower 6 rows) of the matrix
		update_step_pose_arg_stg = K_slam[3*features.shape[1]:,:] @ (features[:,good_obs_inds,i] - z_til).flatten(order='F')
		# split the 6x1 vector obtained above into 2 sets of 3x1 to pass into the hat map for 6-vectors
		update_step_pose_arg_split = np.reshape(update_step_pose_arg_stg, (2,3))
		update_step_pose_arg = hat_map_4(update_step_pose_arg_split[0],update_step_pose_arg_split[1])
		# overwrite the updated_pose variable for i+1
		poses[i+1] = updated_pose @ expm(update_step_pose_arg)

		# covariance update - robot and map done together
		slam_covar = (np.eye(3*features.shape[1]+6,3*features.shape[1]+6) - (K_slam @ H_slam)) @ slam_covar

	fig,ax = plt.subplots()
	ax.plot(poses[:,0,3], poses[:,1,3],color='red')
	ax.scatter(mu_map[0],mu_map[1],s=3)
	# fig.savefig('../plots/d03/slam_noise={x}_stride={y}_outliers_removed={z}_outlier_dist={v}.png'.format(x=str(NOISE_SCALE),y=str(STRIDE),z=str(remove_outliers),v=str(OUTLIER_DIST)))
	# plt.close(fig)

	print('time taken: ', time.time() - start)
	print('# of outliers found during initialization: ', num_outliers_init)