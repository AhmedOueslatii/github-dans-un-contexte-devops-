from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    # Read Video
    video_frames = read_video('input_videos/soccer.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
      ball_bbox = tracks['ball'][frame_num][1]['bbox'] if frame_num < len(tracks['ball']) else None
    
    # Enhanced player assignment logic: prioritize players closer to the ball
      assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox, prioritize_distance=True)

      if assigned_player != -1:
        # Assign the ball to the identified player
        tracks['players'][frame_num][assigned_player]['has_ball'] = True
        team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
      else:
        # Assign to last known team but add a check for proximity to the ball
        if ball_bbox is not None:
            nearest_player_id = player_assigner.get_nearest_player(player_track, ball_bbox)
            if nearest_player_id != -1:
                tracks['players'][frame_num][nearest_player_id]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][nearest_player_id]['team'])
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else None)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else None)

    team_ball_control = np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    if output_video_frames:
     save_video(output_video_frames, 'output_videos/output_video.avi')
    else:
     print("No output frames to save.")

if __name__ == '__main__':
    main()