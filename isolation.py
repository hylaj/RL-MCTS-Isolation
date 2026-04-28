from abc import abstractmethod
from enum import Enum
import itertools
import random
import time
from typing import Optional, Protocol
import math
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Colour(Enum):
    RED = 'R'
    BLUE = 'B'

    def flip(self) -> 'Colour':
        return Colour.RED if self == Colour.BLUE else Colour.BLUE


class Board:
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.positions: dict[tuple[int, int], str] = dict()
        self.red_position: Optional[tuple[int, int]] = None
        self.blue_position: Optional[tuple[int, int]] = None
        self._prepare_board()

    def _prepare_board(self):
        for i in range(self.width):
            for j in range(self.height):
                self.positions[(i, j)] = '.'

    def __str__(self):
        representation = '\\ ' + ' '.join([str(i + 1) for i in range(self.width)]) + '\n'
        for j in range(self.height):
            representation += (chr(ord('A') + j) + ' ' + ' '.join([self.positions[i, j] for i in range(self.width)]))
            if j < self.height - 1:
                representation += '\n'
        return representation

    def moves_for(self, current_player: Colour) -> list[tuple[int, int]]:
        result = []
        player_position = self._player_position(current_player)
        if player_position is None:
            for position in self.positions:
                if self.positions[position] == '.':
                    result.append(position)
        else:
            directions = list(itertools.product([-1, 0, 1], repeat=2))
            directions.remove((0, 0))
            for dx, dy in directions:
                px, py = player_position
                px, py = px + dx, py + dy
                while 0 <= px < self.width and 0 <= py < self.height:
                    potential_position = px, py
                    if self.positions[potential_position] == '.':
                        result.append(potential_position)
                        px, py = px + dx, py + dy
                    else:
                        break
        return result

    def apply_move(self, current_player: Colour, move: tuple[int, int]) -> None:
        player_position = self._player_position(current_player)
        if player_position is not None:
            self.positions[player_position] = '#'
        self.positions[move] = current_player.value
        self._update_player_position(current_player, move)

    def _player_position(self, current_player: Colour) -> tuple[int, int]:
        return self.red_position if current_player == Colour.RED else self.blue_position

    def _update_player_position(self, current_player: Colour, new_position: tuple[int, int]) -> None:
        if current_player == Colour.RED:
            self.red_position = new_position
        else:
            self.blue_position = new_position

    def to_state_str(self) -> str:
        positions_in_order = []
        for j in range(self.height):
            for i in range(self.width):
                positions_in_order.append(self.positions[(i, j)])
        return f"{self.width}_{self.height}_{''.join(positions_in_order)}"

    @staticmethod
    def from_state_str(state_str: str) -> 'Board':
        width, height, positions = state_str.split('_')
        width, height = int(width), int(height)
        board = Board(width, height)
        for j in range(height):
            for i in range(width):
                position = positions[j * width + i]
                board.positions[(i, j)] = position
                if position == Colour.RED.value:
                    board.red_position = (i, j)
                elif position == Colour.BLUE.value:
                    board.blue_position = (i, j)
        return board

    def duplicate(self) -> 'Board':
        return self.from_state_str(self.to_state_str())


class Player(Protocol):
    @abstractmethod
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def register_opponent_action(self, action: tuple[int, int]) -> None:
        raise NotImplementedError


class Game:
    # TODO: tutaj poznasz zasady tego wariantu gry w izolację, są bardzo proste
    # zasady:
    #  * jest dwóch graczy, czerwony i niebieski, czerwony porusza się pierwszy
    #  * każdy gracz ma dokładnie jeden pionek w swoim kolorze ('R' lub 'B')
    #  * plansza jest prostokątem, w swoim pierwszym ruchu każdy gracz może położyć pionek na jej dowolnym pustym polu
    #  * w kolejnych ruchach gracze naprzemiennie przesuwają swoje pionki
    #     * pionki poruszają się jak hetmany szachowe (dowolna liczba pól w poziomie, pionie, lub po skosie)
    #     * pole, z którego pionek startował jest usuwane z planszy ('.' zastępuje '#') i trwale zablokowane
    #     * zarówno pionek innego gracza jak i zablokowane pola uniemożliwiają dalszy ruch (nie da się ich przeskoczyć)
    #  * jeżeli gracz musi wykonać ruch pionkiem, a nie jest to możliwe (każdy z ośmiu kierunków zablokowany)...
    #  * ...to taki gracz przegrywa (a jego przeciwnik wygrywa ;])
    def __init__(self, red: Player, blue: Player, board: Board, current_player: Colour = Colour.RED):
        self.red: Player = red
        self.blue: Player = blue
        self.board: Board = board
        self.current_player: Colour = current_player
        self.finished: bool = False
        self.winner: Optional[Colour] = None

    def run(self, verbose=False):
        if verbose:
            print()
            print(self.board)

        while not self.finished:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) == 0:
                self.finished = True
                self.winner = Colour.BLUE if self.current_player == Colour.RED else Colour.RED
                break

            player = self.red if self.current_player == Colour.RED else self.blue
            opponent = self.red if self.current_player == Colour.BLUE else self.blue
            move = player.choose_action(self.board, self.current_player)
            opponent.register_opponent_action(move)
            self.board.apply_move(self.current_player, move)
            self.current_player = self.current_player.flip()

            if verbose:
                print()
                print(self.board)

        if verbose:
            print()
            print(f"WINNER: {self.winner.value}")


class RandomPlayer(Player):
    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        legal_moves = board.moves_for(current_player)
        return random.sample(legal_moves, 1)[0]

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        pass


class MCTSNode:
    def __init__(self, board: Board, current_player: Colour, c_coefficient: float):
        self.parent: Optional[MCTSNode] = None
        self.leaf: bool = True
        self.terminal: bool = False
        self.times_chosen: int = 0
        self.value: float = 0.5
        self.children: dict[tuple[int, int], MCTSNode] = dict()
        self.board: Board = board
        self.current_player: Colour = current_player
        self.c_coefficient: float = c_coefficient

    def select(self, final=False) -> tuple[int, int]:
        # TODO: tutaj należy wybrać (i zwrócić) najlepszą możliwą akcję (w oparciu o aktualną wiedzę)
        # podpowiedzi:
        #  * klucze w słowniku `self.children` to pula dostępnych akcji
        #  * każdą z nich należy ocenić zgodnie z techniką UCB (tak jakby był to problem wielorękiego bandyty)
        #  * ocena akcji zależy od:
        #     * jej wartościowania (`self.value`)
        #     * oraz tego jak często była wybierana (`self.times_chosen`) w porównaniu z rodzicem
        #     * odpowiednie wartości przechowują węzły-dzieci przyporządkowane w słowniku kluczom-akcjom
        #  * w przypadku kilku akcji o takiej samej ocenie - wybieramy losowo
        #  * gdy stosujemy technikę UCB pierwszeństwo mają akcje, które nie były jeszcze nigdy testowane

        if final:
            max_visits = max(child.times_chosen for child in self.children.values())
            best_actions = [move for move, child in self.children.items() if child.times_chosen == max_visits]
            return random.choice(best_actions)
        else:
            best_ucb = -1
            best_actions = []

            for move, child in self.children.items():
                if child.times_chosen == 0:
                    return move
                else:
                    # ucb = v + c * sqrt(ln(N)/n)
                    child_ucb = child.value + self.c_coefficient * math.sqrt(math.log(self.times_chosen)/child.times_chosen)
                    if child_ucb > best_ucb:
                        best_ucb = child_ucb
                        best_actions = [move]
                    elif child_ucb == best_ucb:
                        best_actions.append(move)
        return random.choice(best_actions)

    def expand(self) -> None:
        if not self.terminal and self.leaf:
            legal_moves = self.board.moves_for(self.current_player)
            if len(legal_moves) > 0:
                self.leaf = False
                oponent = self.current_player.flip()
                for move in legal_moves:
                    child_board = self.board.duplicate()
                    child_board.apply_move(self.current_player, move)
                    child = MCTSNode(child_board, oponent, self.c_coefficient)
                    child.parent = self
                    self.children[move] = child
            else:
                self.terminal = True

    def simulate(self) -> Colour:
        # TODO: w tym węźle rozgrywka nie zakończyła się, więc do ustalenia zwycięzcy potrzebna jest symulacja
        # podpowiedzi:
        #  * w tym celu najłatwiej uruchomić osobną, niezależną grę startującą z danego stanu planszy
        #  * by sumulacja przebiegała możliwe szybko wykonujemy ją z użyciem losowych agentów
        #  * po jej zakończeniu poznajemy i zwracamy zwycięzcę
        if not self.terminal:
            simulated_board = self.board.duplicate()
            red_random_player = RandomPlayer()
            blue_random_player = RandomPlayer()
            simulated_game = Game(red=red_random_player, blue=blue_random_player, board=simulated_board, current_player= self.current_player)

            simulated_game.run(verbose=False)

            return simulated_game.winner
        else:
            return self.current_player.flip()

    def backpropagate(self, winner: Colour) -> None:
        # TODO: należy zaktualizować drzewo - wiedząc, że przejście przez ten węzeł skończyło się wygraną danego gracza
        # podpowiedzi:
        #  * przede wszystkim należy zaktualizować licznik odwiedzeń (`self.times_chosen`)
        #  * poza tym, konieczna jest też korekta wartościowania (`self.value`)
        #     * siła korekty powinna zależeć od tego, które to z kolei odwiedziny danego węzła
        #     * uwaga - fakt, iż np. gracz czerwony wygrał partię ma inny wpływ na wartościowanie jego węzłów...
        #     * ...a inny na wartościowanie węzłów, w których ruch musiał wykonać jego przeciwnik
        #  * warto pamiętać, by po aktualizacji danych węzeł powiadomił o takiej konieczności również swojego rodzica
        self.times_chosen +=1

        if winner != self.current_player:
            prize = 1
        else: prize = 0

        self.value += (prize - self.value)/self.times_chosen

        if self.parent:
            self.parent.backpropagate(winner)



class MCTSPlayer(Player):
    def __init__(self, time_limit: float, c_coefficient: float):
        self.time_limit: float = time_limit
        self.root_node: Optional[MCTSNode] = None
        self.c_coefficient: float = c_coefficient

    def choose_action(self, board: Board, current_player: Colour) -> tuple[int, int]:
        if self.root_node is None:
            self.root_node = MCTSNode(board.duplicate(), current_player, self.c_coefficient)
        start_time = time.time()

        while True:
            self._mcts_iteration()
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= self.time_limit:
                break

        action = self.root_node.select(final=True)  # TODO należy zmienić selekcje tak, by wybrała najlepszą akcję
        # podpowiedź: zamiast UCB wystarczy zwrócić akcję najczęściej odwiedzaną
        self._step_down(action)
        return action

    def register_opponent_action(self, action: tuple[int, int]) -> None:
        if self.root_node is not None:
            self.root_node.expand()
            self._step_down(action)

    def _mcts_iteration(self):
        node = self.root_node
        while not node.leaf:
            action = node.select()
            node = node.children[action]
        node.expand()
        winner = node.simulate()
        node.backpropagate(winner)

    def _step_down(self, action: tuple[int, int]) -> None:
        new_root = self.root_node.children[action]
        new_root.parent = None
        self.root_node = new_root






def _create_player(config: dict):
    if config['type'] == 'MCTS':
        return MCTSPlayer(config['time_limit'], config['c_coefficient'])
    elif config['type'] == 'Random':
        return RandomPlayer()

def _play_single_game_worker(args):

    p1_config, p2_config, width, height, p1_starts = args
    board = Board(width, height)
    
    p1 = _create_player(p1_config)
    p2 = _create_player(p2_config)

    if p1_starts:
        game = Game(p1, p2, board)
        game.run(verbose=False)
        return 1 if game.winner == Colour.RED else 0
    else:
        game = Game(p2, p1, board)
        game.run(verbose=False)
        return 1 if game.winner == Colour.BLUE else 0


def evaluate_agents_parallel(p1_config, p2_config, width: int, height: int, num_games: int) -> float:

    tasks = []
    for i in range(num_games):
        p1_starts = (i % 2 == 0) 
        tasks.append((p1_config, p2_config, width, height, p1_starts))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_play_single_game_worker, tasks))
        
    p1_wins = sum(results)
    return p1_wins / num_games



def experiment_time_limits():
    times = [0.02, 0.05, 0.1, 0.2, 0.5]
    baseline_time = 0.05
    c_coeff = 0.5
    board_w, board_h = 5, 4
    num_games = 20 
    win_rates = []
    
    p2_config = {'type': 'MCTS', 'time_limit': baseline_time, 'c_coefficient': c_coeff}

    for t in times:
        p1_config = {'type': 'MCTS', 'time_limit': t, 'c_coefficient': c_coeff}
        wr = evaluate_agents_parallel(p1_config, p2_config, board_w, board_h, num_games)
        win_rates.append(wr)

    plt.figure(figsize=(8, 5))
    plt.plot(times, win_rates)
    plt.axhline(y=0.5)
    plt.title("Wynik MCTS w zależności od czasu na ruch")
    plt.xlabel("time limit (s)")
    plt.ylabel("Win rate")
    plt.grid(True)
    plt.legend()
    plt.savefig("time_limits.png", bbox_inches='tight', dpi=300)

def experiment_c_coefficient():
    c_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    time_limit = 0.1
    board_w, board_h = 4, 4
    num_games = 20 
    
    results = np.zeros((len(c_values), len(c_values)))
    
    for i, c1 in enumerate(c_values):
        for j, c2 in enumerate(c_values):
            if i == j:
                results[i, j] = 0.5 
            else:
                p1_config = {'type': 'MCTS', 'time_limit': time_limit, 'c_coefficient': c1}
                p2_config = {'type': 'MCTS', 'time_limit': time_limit, 'c_coefficient': c2}
                results[i, j] = evaluate_agents_parallel(p1_config, p2_config, board_w, board_h, num_games)

    plt.figure(figsize=(7, 6))
    sns.heatmap(results, annot=True, cmap="coolwarm", center=0.5, 
                xticklabels=c_values, yticklabels=c_values, vmin=0, vmax=1)
    plt.title(f"Heatmap Win Rate\n(time={time_limit}s)")
    plt.xlabel("c_coefficient Player 2")
    plt.ylabel("c_coefficient Player 1")
    plt.savefig("c_coefficient_heatmap.png", bbox_inches='tight', dpi=300)


def experiment_board_size():
    sizes = [(3, 3), (7, 5), (10, 8)]
    time_limit = 0.1
    c_coeff = 0.5
    
    num_games = 40 
    
    win_rates = []
    board_areas = []
    
    p1_config = {'type': 'MCTS', 'time_limit': time_limit, 'c_coefficient': c_coeff}
    p2_config = {'type': 'Random'}

    for w, h in sizes:
        
        wr = evaluate_agents_parallel(p1_config, p2_config, w, h, num_games)
        win_rates.append(wr)
        board_areas.append(f"{w}x{h}")

    plt.figure(figsize=(8, 5))
    plt.bar(board_areas, win_rates, color='g')
    plt.title(f"Board Size impact on the MCTS win rate\n(time={time_limit}s, c={c_coeff})")
    plt.xlabel("Board Size")
    plt.ylabel("Win Rate MCTS")
    plt.ylim(0, 1.05)
    for i, v in enumerate(win_rates):
        plt.text(i, v + 0.02, str(round(v, 2)), ha='center')
    plt.savefig("board_size.png", bbox_inches='tight', dpi=300)


def main() -> None:

    experiment_time_limits()
    experiment_c_coefficient()
    experiment_board_size()


if __name__ == '__main__':
    main()



# def main() -> None:
#     red_wins = 0
#     blue_wins = 0

#     for _ in range(100):
#         board = Board(15, 10)  # (7, 5) TODO: na początek możesz skorzystać z mniejszej planszy (np. 4x4)
#         red_player = MCTSPlayer(0.2, 0.5) # TODO: zastąp jednego z agentów wariantem MCTS
#         # podpowiedź: np. takim `red_player = MCTSPlayer(0.2, 0.5)`
#         blue_player = RandomPlayer()
#         game = Game(red_player, blue_player, board)
#         game.run(verbose=True)  # TODO: jeżeli nie chcesz czytać na konsoli zapisu partii, skorzystaj z `verbose=False`

#         if game.winner == Colour.RED:
#             red_wins += 1
#         else:
#             blue_wins += 1

#     print(red_wins, blue_wins)  # TODO: jeżeli wszystko poszło dobrze, to agent MCTS powtarzalnie wygrywa z losowym


# if __name__ == '__main__':
#     main()  # TODO: jeżeli podstawowy eksperyment zakończył się sukcesem to sprawdź inne jego warianty
#     # podpowiedź:
#     #  * możesz zorganizować pojedynek agentów MCTS o różnych parametrach (np. czasie na wybór akcji)
#     #  * możesz też zmienić rozmiar planszy lub skłonność do eksplorowania (`self.c_coefficient`)
