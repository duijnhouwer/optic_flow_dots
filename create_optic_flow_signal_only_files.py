# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:29:30 2024

@author: jduij
"""
import numpy as np
import optic_flow_dots_signal_only
import torch
import os
import inspect
import time
import the_luggage as lgg
from multiprocessing import Pool
import keyboard


def create_optic_flow_files(n: int=100000,
                            trans_speed_minmax: np.array=np.array([0.03,0.03]),
                            rot_dpf_minmax: np.array=np.array([0,0]),
                            savedir: str='F:\optic_flow_dots_data'):
    start_second=time.time()
    for i in range(n):
        # Pick a rotation and translation speed
        trans_speed = np.min(trans_speed_minmax) + np.random.sample() * trans_speed_minmax.ptp()
        rot_dpf = np.min(rot_dpf_minmax) + np.random.sample() * rot_dpf_minmax.ptp()

        # Create the translation and rotation vectors
        T_ruf = optic_flow_dots_signal_only.random_unit_vector(3) * trans_speed # ruf: right up front
        R_ruf = optic_flow_dots_signal_only.random_unit_vector(3) * rot_dpf

        # Create the Frame-Height-Width tensor movie
        signal = optic_flow_dots_signal_only.generate_signal_array(6000,T_ruf,R_ruf)

        # save to a file with the 6DOF parameters in the filename
        filename = "pico_y=[{:+.0f},{:+.0f},{:+.0f},{:+.0f},{:+.0f},{:+.0f}].pt".format(*T_ruf*1e12,*R_ruf*1e12)
        filename = os.path.join(savedir, filename)
        try:
            torch.save(torch.from_numpy(signal).float(), filename)
        except:
            raise Exception(f"Could not save '{filename}' for unknown reason. Maybe another process was writing a file with the same name at the same time? No biggie, just one of a million")

        if i%1000==0:
            print("[{}] Saved {} movies to {} (Elapsed time={})".format(inspect.currentframe().f_code.co_name,i,savedir,lgg.format_duration(time.time()-start_second)))

    print("[{}] Done".format(inspect.currentframe().f_code.co_name))


def main():

    n_workers = 8
    if n_workers <= 0:
        # For testing
        create_optic_flow_files(10)
    else:
        try:
            lgg.computer_sleep('disable')
            print(f"Starting pool of {n_workers} workers.")
            with Pool(n_workers) as p:
                print("Press and hold ESC-key to cancel early.")
                p.imap(create_optic_flow_files, [1000000] *n_workers)
                while True:
                    time.sleep(2)
                    if keyboard.is_pressed('esc'):
                        print("ESC-key pressed.")
                        break
        finally:
            lgg.computer_sleep('enable')
            print(f"Sending terminate signal to all {n_workers} workers ...")
            p.terminate()
            p.join()
    print('[-: The End :-]')


if __name__=="__main__":
    main()
