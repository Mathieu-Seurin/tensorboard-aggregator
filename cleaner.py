import os
import tensorflow as tf

import shutil

expe_path = '.'

deleted_120 = 0
deleted_100 = 0

deleted = 0

total = 0

for dir in os.listdir(expe_path):

    expe_sub_dir = os.path.join(expe_path, dir)

    if os.path.isdir(expe_sub_dir):

        if os.path.isfile(os.path.join(expe_sub_dir, "42", "best_model.pth")):
            expe_sub_dir = os.path.join(expe_sub_dir, "42")

        for file in os.listdir(expe_sub_dir):

            if "tfevents" in file:

                total += 1

                last_iter = 0

                tf_event_path = os.path.join(expe_sub_dir, file)

                for elem in tf.train.summary_iterator(tf_event_path) :
                    if elem.step:
                        last_iter = max(last_iter, elem.step)

                to_delete = False
                for elem in tf.train.summary_iterator(tf_event_path):
                    if elem.step and elem.summary:
                        if elem.step == last_iter and "reward_wo_feedback_unbiaised" in elem.summary.value[0].tag:
                            if elem.summary.value[0].simple_value < 0.90:
                                to_delete = True


                if to_delete or last_iter < 10000:
                    print(expe_sub_dir[:-3])
                    shutil.rmtree(expe_sub_dir[:-3])
                    deleted += 1

                # if last_iter > 120000:
                #     #print(expe_sub_dir)
                #     deleted_120 += 1
                #     #os.remove(expe_sub_dir)
                #
                # if last_iter > 100000:
                #     deleted_100 += 1
                #     os.remove(tf_event_path)



print("Total = ", total)
print("Deleted =", deleted)
# print("Deleted 120 = ", deleted_120)
# print("Deleted 100 = ", deleted_100)