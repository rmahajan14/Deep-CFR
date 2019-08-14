from PokerRL.game.games import StandardLeduc, DiscretizedNLHoldem  # or any other game
from PokerRL.eval.lbr import LBRArgs
from PokerRL.game import bet_sets
from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver
from PokerRL import DiscretizedNLHoldem, Poker

if __name__ == '__main__':
    N_LBR_WORKERS = 40
    ctrl = Driver(t_prof=TrainingProfile(name="Hanul_EXAMPLE",
                                         nn_type="recurrent",
                                         max_buffer_size_adv=3e6,
                                         eval_agent_export_freq=20,  # export API to play against the agent
                                         n_traversals_per_iter=200,
                                         n_batches_adv_training=8,
                                         n_batches_avrg_training=2000,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_adv=64,
                                         n_units_final_avrg=64,
                                         mini_batch_size_adv=16,
                                         mini_batch_size_avrg=16,
                                         init_adv_model="last",
                                         init_avrg_model="last",
                                         use_pre_layers_adv=False,
                                         use_pre_layers_avrg=False,

                                         game_cls = DiscretizedNLHoldem,
										 
										 lbr_args=LBRArgs(
                                             lbr_bet_set=bet_sets.B_5,
                                             n_lbr_hands_per_seat=80,
                                             lbr_check_to_round=Poker.TURN,
                                             n_parallel_lbr_workers=N_LBR_WORKERS,
                                             use_gpu_for_batch_eval=False,
                                             DISTRIBUTED=True,
                                         ),

                                         # You can specify one or both modes. Choosing both is useful to compare them.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # SD-CFR
                                             EvalAgentDeepCFR.EVAL_MODE_AVRG_NET,  # Deep CFR
                                         ),

                                         DISTRIBUTED=False,
                                         ),
                  eval_methods={
                      "lbr": 5,
                  },
                  n_iterations=300)
    ctrl.run()