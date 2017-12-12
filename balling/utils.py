import logging
import os
import subprocess

from balling import input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_with_logging(cmd):
    """
    Run cmd and wait for it to finish. While cmd is running, we read it's
    output and print it to a logger.

    :param cmd: command to run. Should be a list of command tokens, e.g. ['ls', '-l']
    """
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = process.stdout.readline()
        if not line:
            break
        logger.info(line.rstrip('\n'))

    exit_code = process.wait()

    if exit_code:
        raise subprocess.CalledProcessError(exit_code, cmd)

    return exit_code


def split_audio_files(path, outpath, extension=None, segment_duration=12):
    """
    Splits audio files into time segments of seconds. The output files are named after the input file, but save
    in a specified directory

    :param path: path to input files. Can be a directory or a single file
    :param outpath: directory where output file will be written to
    :param segment_duration: duration of each segment in seconds
    :return:
    """

    if extension and not extension.endswith('.'):
        extension = ''.join(['.', extension])

    if os.path.isfile(path):
        files = [path]
    else:
        files, _ = input.filenames_and_labels_in_dir(path, extension)

    part_suffix = '%06d'

    for filepath in files:
        if extension:
            outfile = os.path.join(
                outpath,
                ''.join([os.path.basename(filepath[0:-len(extension)]), part_suffix, extension])
            )

        else:
            outfile = os.path.join(
                outpath,
                ''.join([os.path.basename(filepath), part_suffix])
            )

        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))

        cmd = [
            'ffmpeg',
            '-i',
            filepath,
            '-f',
            'segment',
            '-segment_time',
            str(segment_duration),
            '-c',
            'copy',
            outfile
        ]

        run_with_logging(cmd)
