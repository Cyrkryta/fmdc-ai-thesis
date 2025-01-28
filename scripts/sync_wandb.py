import getpass
import logging
import os
import shutil
import subprocess
from glob import glob
from time import sleep

remote_dir = 'q038jt@titan.i-med.ac.at:/home/j/jt/038/q038jt/field-map-ai/wandb'
mount_dir = '/tmp/sshmount'


def _sync_runs():
    try:
        subprocess.check_output(f'wandb sync --project field-map-ai {mount_dir}/offline-run-*', shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error(f'Syncing offline runs failed: {e.output}')

    logging.info('Done syncing runs!')


def _remove_old_runs():
    runs = glob(f'{mount_dir}/offline-run-*/')
    runs.sort()
    if len(runs) > 1:
        logging.info(f'Removing {len(runs) - 1} old runs...')
        for run in runs[:-1]:
            shutil.rmtree(run)
        logging.info('Done removing old runs!')


def main():
    password = getpass.getpass(prompt='Password for remote server:')

    if not os.path.exists(mount_dir):
        os.makedirs(mount_dir)

    try:
        mount_process = subprocess.Popen(['sshfs', '-o', 'password_stdin', remote_dir, mount_dir], stdin=subprocess.PIPE, shell=False, text=True)
        mount_process.communicate(input=password)
    except subprocess.CalledProcessError as e:
        logging.error(f'Mounting the SSH directory failed: {e.output}')

    while True:
        sleep(20)
        logging.info('Syncing offline runs...')
        _sync_runs()
        _remove_old_runs()


if __name__ == '__main__':
    """
    Quick helper script for manually syncing wandb runs from the GPU server since it is firewalled and cannot do that 
    by itself.
    """

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    try:
        main()
    except KeyboardInterrupt as e:
        logging.info('Stopping synchronization!')
    finally:
        logging.info('Stopping SSH mount process...')
        try:
            subprocess.check_output(['killall', '-9', 'sshfs'], shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.error(f'Killing the mount process failed: {e.output}')

        logging.info('Unmounting...')
        try:
            subprocess.check_output(['umount', mount_dir], shell=False, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            logging.error(f'Unmounting the SSH directory failed: {e.output}')
