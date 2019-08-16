# Copyright (c) 2019 Eric Steinberger


"""
This file is not runable; it's is a template to show how you could play against your algorithms. To do so,
replace "YourAlgorithmsEvalAgentCls" with the EvalAgent subclass (not instance) of your algorithm.
Note that you can see the AI's cards on the screen since this is just a research application and not meant for actual
competition. The AI can, of course, NOT see your cards.
"""
import pickle
from PokerRL.game.InteractiveGame import InteractiveGame

if __name__ == '__main__':
#a = open(r"C:\Users\NiFa\Desktop\Data Science\Deep Learning\deep_cfr_github\poker_interactive\trained_agents\Example_FHP_AVRG_NET.pkl","rb")
#    a = open(r"C:\Users\NiFa\Desktop\Data Science\Deep Learning\deep_cfr_github\poker_interactive\single_1_15.0.pkl","rb")
    a = open(r"C:\Users\ridhi\Deep-CFR\single_0_0.0.pkl","rb")
    eval_agent = pickle.load(a)
    
    game = InteractiveGame(env_cls=eval_agent.env_bldr.env_cls,
                           env_args=eval_agent.env_bldr.env_args,
                           seats_human_plays_list=[0],
                           eval_agent=eval_agent,
                           )
    
    game.start_to_play()