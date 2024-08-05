import ahrs
import matplotlib.pyplot as plt
import numpy as np

from ..data import event_data


def plot_sensors(sensor_set, axis=0, data_type="gyro"):
    no_sensors = len(sensor_set.sensor_data)
    fig, axes = plt.subplots(no_sensors)
    for sensor_idx in range(0, no_sensors):
        sensor = sensor_set.sensor_data[sensor_idx]
        if data_type == "gyro":
            raw_data = sensor.gyro_data
        elif data_type == "acc":
            raw_data = sensor.acc_data
        elif data_type == "mag":
            raw_data = sensor.mag_data
        else:
            raise ValueError("Data type not recognized")
        t = np.arange(raw_data.shape[0])
        if no_sensors > 1:
            ax = axes[sensor_idx]
        else:
            ax = axes
        ax.plot(t, raw_data[:, axis], label='gyro')

        # Plot events if we have gait cycles
        # TODO: this assumes first HS is right, otherwise labels should change
        if len(sensor_set.gait_cycles) > 0:
            invalid_events = [event.sample_idx for event in sensor_set.events
                              if event.validity != event_data.GaitEventValidity.VALID]
            rhs_idx = [step.hs_start.sample_idx for step
                       in sensor_set.gait_cycles]
            rhs_idx.extend([step.hs_end.sample_idx for step
                            in sensor_set.gait_cycles])
            rto_idx = [step.to.sample_idx for step in sensor_set.gait_cycles]
            lhs_idx = [step.hs_opposite.sample_idx for step
                       in sensor_set.gait_cycles]
            lto_idx = [step.to_opposite.sample_idx for step
                       in sensor_set.gait_cycles]
            ax.plot(rhs_idx, raw_data[:, axis][rhs_idx], 'r*', label='HS_r')
            ax.plot(lto_idx, raw_data[:, axis][lto_idx], 'g*', label='TO_l')
            ax.plot(lhs_idx, raw_data[:, axis][lhs_idx], 'm*', label='HS_l')
            ax.plot(rto_idx, raw_data[:, axis][rto_idx], 'c*', label='TO_r')
            ax.plot(invalid_events, raw_data[:, axis][invalid_events], 'ko',
                    label='invalid', alpha=0.5)

        title = f"Sensor {sensor_idx+1} at position {sensor.sensor_position}"
        ax.set_title(title)
        ax.legend(loc="upper right")


# Plot events
def plot_wavelet_events(data, acc_ap_pp, acc_wave_detrended, acc_wave_2):
    time_vector = data.time_vector
    hs_idx = [event.sample_idx for event in data.events
              if event.event_type == event_data.GaitEventType.HEEL_STRIKE]
    to_idx = [event.sample_idx for event in data.events
              if event.event_type == event_data.GaitEventType.TOE_OFF]
    inval_hs_idx = [event.sample_idx for event in data.events
                    if (event.event_type == event_data.GaitEventType.HEEL_STRIKE
                        and event.validity != event_data.GaitEventValidity.VALID)]
    hs_times = time_vector[hs_idx]
    to_times = time_vector[to_idx]
    inval_hs_times = time_vector[inval_hs_idx]
    plt.figure(0)
    plt.plot(time_vector, acc_ap_pp, 'b--', label='acc_ap')
    plt.plot(time_vector, acc_wave_detrended, color="grey",
             linestyle="dotted", label='dcwt1')
    plt.plot(time_vector, acc_wave_2, 'k', label='dcwt2')
    plt.plot(hs_times, acc_wave_detrended[hs_idx],
             linestyle='None', color='red', label='HS',
             marker="*",  markersize=10)
    plt.plot(to_times, acc_wave_2[to_idx],
             linestyle='None', color='lime', label='TO',
             marker="*",  markersize=10)
    plt.plot(inval_hs_times, acc_wave_detrended[inval_hs_idx],
             linestyle='None', color='orange', label='HS invalid',
             marker="*",  markersize=10)
    plt.xlabel('Time [s]')
    plt.legend(loc="upper right")
    plt.show()


def plot_gyro_detection(data, side, gyro_data):
    time_vector = data.time_vector
    hs_events = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                                side=side)
    hs = [event.sample_idx for event in hs_events]
    to_events = data.get_events(event_data.GaitEventType.TOE_OFF,
                                side=side)
    to = [event.sample_idx for event in to_events]
    ms_events = data.get_events(event_data.GaitEventType.MID_SWING,
                                side=side)
    ms = [event.sample_idx for event in ms_events]
    ff_events = data.get_events(event_data.GaitEventType.FOOT_FLAT,
                                side=side)
    ff = [event.sample_idx for event in ff_events]
    plt.plot(time_vector, gyro_data, 'b', label='Gyro')
    plt.plot(time_vector[hs], gyro_data[hs], linestyle='None',
             color='red', label='HS', marker="*",  markersize=10)
    plt.plot(time_vector[to], gyro_data[to], linestyle='None',
             color='lime', label='TO', marker="*",  markersize=10)
    plt.plot(time_vector[ff], gyro_data[ff], linestyle='None',
             color='black', label='FF', marker="*",  markersize=10)
    plt.plot(time_vector[ms], gyro_data[ms], linestyle='None',
             color='yellow', label='MS', marker="*",  markersize=10)
    plt.xlabel('Time [s]')
    plt.legend(loc="upper right")
    plt.show()


def plot_position(data, events=None):
    # Get events to calculate positions between
    all_event_idx = [event.sample_idx for event in events]
    plt.figure(0)
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1, 1, 1])
    ax.plot3D(data.position[:, 0], data.position[:, 1], data.position[:, 2],
              'gray')
    ax.plot3D(data.position[:, 0][all_event_idx],
              data.position[:, 1][all_event_idx],
              data.position[:, 2][all_event_idx], 'k*')
    ax.set_xlabel('X (East)')
    ax.set_ylabel('Y (North)')
    ax.set_zlabel('Z (Up)')
    set_axes_equal(ax)
    plt.show()


def plot_velocity(data):
    vel = data.velocity
    plt.figure(0)
    t = np.arange(vel.shape[0])
    plt.plot(t, vel[:, 0], label='x')
    plt.plot(t, vel[:, 1], label='y')
    plt.plot(t, vel[:, 2], label='z')
    plt.title('velocity')
    plt.legend()
    plt.show()


# Plot two data with HS
def plot_axes_with_hs(data, data_1, data_2, label):
    hs_lf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.LEFT)
    hs_lf = [event.sample_idx for event in hs_lf]
    hs_rf = data.get_events(event_data.GaitEventType.HEEL_STRIKE,
                            event_data.GaitEventSide.RIGHT)
    hs_rf = [event.sample_idx for event in hs_rf]
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(data_1.shape[0])
    ax1.plot(t, data_1[:, 0], label='X 1')
    ax1.plot(t, data_2[:, 0], label='X 2')
    ax1.plot(hs_lf, data_2[:, 0][hs_lf], 'r*', label='HS lf')
    ax1.plot(hs_rf, data_2[:, 0][hs_rf], 'c*', label='HS rf')
    ax1.legend(loc="upper right")
    ax1.set_title(label + ' X')
    ax2.plot(t, data_1[:, 1], label='Y 1')
    ax2.plot(t, data_2[:, 1], label='Y 2')
    ax2.plot(hs_lf, data_2[:, 1][hs_lf], 'r*', label='HS lf')
    ax2.plot(hs_rf, data_2[:, 1][hs_rf], 'c*', label='HS rf')
    ax2.legend(loc="upper right")
    ax2.set_title(label + ' Y')
    ax3.plot(t, data_1[:, 2], label='Z 1')
    ax3.plot(t, data_2[:, 2], label='Z 2')
    ax3.plot(hs_lf, data_2[:, 2][hs_lf], 'r*', label='HS lf')
    ax3.plot(hs_rf, data_2[:, 2][hs_rf], 'c*', label='HS rf')
    ax3.legend(loc="upper right")
    ax3.set_title(label + ' Z')


# Plot a quaternion
def plot_quaternions_euler(q, ff):
    fig, (ax1, ax2) = plt.subplots(2)

    # Quaternions
    t = np.arange(q.shape[0])
    ax1.plot(t, q[:, 0], label='w')
    ax1.plot(t, q[:, 1], label='x')
    ax1.plot(t, q[:, 2], label='y')
    ax1.plot(t, q[:, 3], label='z')
    ax1.plot(ff, q[:, 0][ff], 'k*')
    ax1.plot(ff, q[:, 1][ff], 'k*')
    ax1.plot(ff, q[:, 2][ff], 'k*')
    ax1.plot(ff, q[:, 3][ff], 'k*')
    ax1.set_xlabel('Sample', fontsize=15)
    ax1.set_ylim(-1, 1)
    ax1.set_title("Quaternion", size=15)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.tick_params(axis='both', which='minor', labelsize=13)
    ax1.legend(loc="upper right")

    # Euler angles
    euler_angles = np.array(
        [ahrs.Quaternion(q_arr).to_angles() for q_arr in q])
    euler_angles = np.degrees(euler_angles)
    t = np.arange(euler_angles.shape[0])
    ax2.plot(t, euler_angles[:, 0], label='x')
    ax2.plot(t, euler_angles[:, 1], label='y')
    ax2.plot(t, euler_angles[:, 2], label='z')
    ax2.plot(ff, euler_angles[:, 0][ff], 'k*')
    ax2.plot(ff, euler_angles[:, 1][ff], 'k*')
    ax2.plot(ff, euler_angles[:, 2][ff], 'k*')
    ax2.set_xlabel('Sample', fontsize=15)
    ax2.set_title("Euler", size=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='minor', labelsize=13)
    ax2.legend(loc="upper right")
    plt.show()


def plot_profile(ff, sensor_data, titlestr):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    for ff_idx in range(0, len(ff) - 1):
        tff = ff[ff_idx]
        next_tff = ff[ff_idx+1]
        if tff == next_tff:
            continue
        data_interval = sensor_data[tff: next_tff, :]
        t = np.arange(data_interval.shape[0])
        ax1.plot(t, data_interval[:, 0])
        ax2.plot(t, data_interval[:, 1])
        ax3.plot(t, data_interval[:, 2])
    ax1.set_title(f"{titlestr}: X")
    ax2.set_title(f"{titlestr}: Y")
    ax3.set_title(f"{titlestr}: Z")
    plt.show()


def plot_filter(gyro_data, acc_data, mag_data,
                gyro_data_filt, acc_data_filt, mag_data_filt):

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(gyro_data.shape[0])
    ax1.plot(t, gyro_data[:, 0], label='X orig')
    ax1.plot(t, gyro_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Gyro X filt')
    ax2.plot(t, gyro_data[:, 1], label='Y orig')
    ax2.plot(t, gyro_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Gyro Y filt')
    ax3.plot(t, gyro_data[:, 2], label='Z orig')
    ax3.plot(t, gyro_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Gyro Z filt')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(acc_data.shape[0])
    ax1.plot(t, acc_data[:, 0], label='X orig')
    ax1.plot(t, acc_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Acc X filt')
    ax2.plot(t, acc_data[:, 1], label='Y orig')
    ax2.plot(t, acc_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Acc Y filt')
    ax3.plot(t, acc_data[:, 2], label='Z orig')
    ax3.plot(t, acc_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Acc Z filt')
    plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    t = np.arange(mag_data.shape[0])
    ax1.plot(t, mag_data[:, 0], label='X orig')
    ax1.plot(t, mag_data_filt[:, 0], label='X filt')
    ax1.legend(loc="upper right")
    ax1.set_title('Mag X filt')
    ax2.plot(t, mag_data[:, 1], label='Y orig')
    ax2.plot(t, mag_data_filt[:, 1], label='Y filt')
    ax2.legend(loc="upper right")
    ax2.set_title('Mag Y filt')
    ax3.plot(t, mag_data[:, 2], label='Z orig')
    ax3.plot(t, mag_data_filt[:, 2], label='Z filt')
    ax3.legend(loc="upper right")
    ax3.set_title('Mag Z filt')
    plt.show()


# Helper function for plotting 3D
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
