from PokerRL.game.games import StandardLeduc  # or any other game

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

from PokerRL import DiscretizedNLHoldem, Poker
from PokerRL.eval.lbr import LBRArgs
from PokerRL.game import bet_sets

if __name__ == '__main__':
    N_WORKERS = 16
    N_LBR_WORKERS = 3
    ctrl = Driver(t_prof=TrainingProfile(
#                                            name="SD-CFR_LEDUC_EXAMPLE",
#                                         nn_type="feedforward",
#                                         n_learner_actor_workers=N_WORKERS,
#                                         max_buffer_size_adv=3e6,
#                                         eval_agent_export_freq=20,  # export API to play against the agent
#                                         n_traversals_per_iter=1500,
#                                         n_batches_adv_training=750,
#                                         n_batches_avrg_training=2000,
#                                         n_merge_and_table_layer_units_adv=64,
#                                         n_merge_and_table_layer_units_avrg=64,
#                                         n_units_final_adv=64,
#                                         n_units_final_avrg=64,
#                                         mini_batch_size_adv=2048,
#                                         mini_batch_size_avrg=2048,
#                                         init_adv_model="last",
#                                         init_avrg_model="last",
#                                         use_pre_layers_adv=False,
#                                         use_pre_layers_avrg=False,

                                         eval_agent_export_freq=9999999,  # Don't export
                                         
                                         max_buffer_size_adv=3.636e5,  # 364k * 11 = ~4M
                                         max_buffer_size_avrg=3.636e5,  # 364k * 11 = ~4M
                                         
                                         n_traversals_per_iter=50,  # 800 * 11 = 8,800
                                         
                                         n_batches_adv_training=12,
                                         n_batches_avrg_training=100,  # trained far more than necessary
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         n_cards_state_units_adv=64,
                                         n_cards_state_units_avrg=64,
                                         mini_batch_size_adv=8,  # 256 * 11 = 2,816
                                         mini_batch_size_avrg=16,  # 512 * 11 = 5,632
                                         init_adv_model="last",  # warm start neural weights with init from last iter
                                         init_avrg_model="random",
                                         use_pre_layers_adv=True,
                                         use_pre_layers_avrg=True,
                                         
                                         
                                         
                                         
                                         name="NFSP_DISTRIBUTED_LH_RNN",

                                         DISTRIBUTED=True,
                                         n_learner_actor_workers=N_WORKERS,

                                         nn_type="recurrent",

                                         game_cls=DiscretizedNLHoldem,
                                         agent_bet_set=bet_sets.B_5,

#                                         use_pre_layers_br=True,
#                                         use_pre_layers_avg=True,
#                                         n_units_final_br=64,
#                                         n_units_final_avg=64,
#                                         n_merge_and_table_layer_units_br=64,
#                                         n_merge_and_table_layer_units_avg=64,
#                                         rnn_units_br=64,
#                                         rnn_units_avg=64,
#                                         n_cards_state_units_br=128,
#                                         n_cards_state_units_avg=128,
#                                         
#                                         cir_buf_size_each_la=6e5 / N_WORKERS,
#                                         res_buf_size_each_la=2e6,
#                                         n_envs=128,
#                                         n_steps_per_iter_per_la=128,
#
#                                         lr_br=0.1,
#                                         lr_avg=0.01,
#
#                                         mini_batch_size_br_per_la=64,
#                                         mini_batch_size_avg_per_la=64,
#                                         n_br_updates_per_iter=1,
#                                         n_avg_updates_per_iter=1,
#
#                                         eps_start=0.08,
#                                         eps_const=0.007,
#                                         eps_exponent=0.5,
#                                         eps_min=0.0,

                                         lbr_args=LBRArgs(
                                             lbr_bet_set=bet_sets.B_5,
                                             n_lbr_hands_per_seat=50,
                                             lbr_check_to_round=Poker.TURN,
                                             n_parallel_lbr_workers=N_LBR_WORKERS,
                                             use_gpu_for_batch_eval=False,
                                             DISTRIBUTED=True,
                                         )
                                         ),
                  eval_methods={"lbr": 50},
                  n_iterations=None)
    ctrl.run()
