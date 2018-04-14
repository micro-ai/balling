import sys
import itertools
import os

ping_pong_files_dir = sys.argv[1]
noise_files_dir = sys.argv[2]
out_dir = sys.argv[3]

for ping_pong_file, noise_file in itertools.product(os.listdir(ping_pong_files_dir), os.listdir(noise_files_dir)):

    ffmpeg_command = 'ffmpeg -i {pingis} -filter_complex "amovie={noise}:loop=999[s];[0][s]amix=duration=shortest" {output}'.format(
        pingis=os.path.join(ping_pong_files_dir, ping_pong_file),
        noise=os.path.join(noise_files_dir, noise_file),
        output="{}/{}-{}.wav".format(ping_pong_files_dir, ping_pong_file.strip('.wav'), noise_file))

    os.system(ffmpeg_command)
