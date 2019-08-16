# Copyright (c) 2019 Eric Steinberger


"""
This file is not runable; it's is a template to show how you could play against your algorithms. To do so,
replace "YourAlgorithmsEvalAgentCls" with the EvalAgent subclass (not instance) of your algorithm.
Note that you can see the AI's cards on the screen since this is just a research application and not meant for actual
competition. The AI can, of course, NOT see your cards.
"""
import pickle
import itertools
from PokerRL.game.AgentTournament import AgentTournament


def run_interact(a, b, n_games_per_seat):
    agent1 = open(fr"C:\Users\NiFa\Desktop\Data Science\Deep Learning\\"
                  fr"deep_cfr_github\poker_interactive\change 64 to 128 final layer\trained agents\\"
                  fr"single_1_{a}.0.pkl","rb")
    eval_agent1 = pickle.load(agent1)
    
    agent2 = open(fr"C:\Users\NiFa\Desktop\Data Science\Deep Learning\\"
                  fr"deep_cfr_github\poker_interactive\change 64 to 128 final layer\trained agents\\"
                  fr"single_1_{b}.0.pkl","rb")
    eval_agent2 = pickle.load(agent2)
    
    game = AgentTournament(env_cls=eval_agent1.env_bldr.env_cls,
                           env_args=eval_agent1.env_bldr.env_args,
                           eval_agent_1=eval_agent1,
                           eval_agent_2=eval_agent2,
                           )
    
    print(f'PLAYING {a} against {b}')
    mean, upper_conf95, lower_conf95 = game.run(n_games_per_seat=n_games_per_seat)
    conf_95 = upper_conf95 - lower_conf95
    return mean, conf_95

    
    
    
if __name__ == '__main__':
#a = open(r"C:\Users\NiFa\Desktop\Data Science\Deep Learning\deep_cfr_github\poker_interactive\trained_agents\Example_FHP_AVRG_NET.pkl","rb")
    
    n_games_per_seat = 3000
    i_list = range(0,10)
    d = {}
    for a, b in itertools.product(i_list, i_list):
        mean, conf_95 = run_interact(a, b, n_games_per_seat)
        d[(a, b)] = mean
    path = 'DICT_Iterations_agents_128.pkl'
    with open(path, "wb") as pkl_file:
        pickle.dump(obj=d, file=pkl_file)
#    pickle.dump(d, 'Iterations_agents.pkl')

