from alpha_zero.Coach import Coach
from alpha_zero.NNetWrapper import NNetWrapper
from alpha_zero.utils import dotdict
from envs.stc.stc_game import StcGame

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './logs/alpha_zero/stc/checkpoint',
    'load_model': False,
    'load_folder_file': ('./logs/alpha_zero/stc/models', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})


def main():
    g = StcGame()
    nnet = NNetWrapper(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()


if __name__ == '__main__':
    main()
