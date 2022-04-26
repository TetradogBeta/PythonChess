from os.path import exists
from git import Repo;
import tensorflow as tf;
from operator import itemgetter;
from collections import OrderedDict;
import numpy as np;
import chess;
import pandas as pd;

class ChessModel:
    DefaultPath='chessmodel';

    _pawn_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                            [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                            [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
                            [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                            [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                            [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float)

    _pawn_black_eval = _pawn_white_eval[::-1]


    _knight_white_eval = np.array([[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                                [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
                                [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
                                [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
                                [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
                                [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
                                [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
                                [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]], np.float)

    _knight_black_eval = _knight_white_eval[::-1]


    _bishop_white_eval = np.array([[-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
                                [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
                                [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
                                [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                                [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
                                [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]], np.float)

    _bishop_black_eval = _bishop_white_eval[::-1]


    _rook_white_eval = np.array([[0.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
                                [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                                [ 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]], np.float)

    _rook_black_eval = _rook_white_eval[::-1]


    _queen_white_eval = np.array([[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                                [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                                [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                                [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                                [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                                [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                                [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]], np.float)

    _queen_black_eval = _queen_white_eval[::-1]


    _king_white_eval = np.array([[-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                                [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                                [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                                [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                                [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]], np.float)

    _king_black_eval = _king_white_eval[::-1]

    _square_coord={0:(7,0), 1:(7,1), 2:(7,2), 3:(7,3), 4:(7,4), 5:(7,5), 6:(7,6), 7:(7,7),
                8:(6,0), 9:(6,1), 10:(6,2), 11:(6,3), 12:(6,4), 13:(6,5), 14:(6,6), 15:(6,7), 
                16:(5,0), 17:(5,1), 18:(5,2), 19:(5,3), 20:(5,4), 21:(5,5), 22:(5,6), 23:(5,7),
                24:(4,0), 25:(4,1), 26:(4,2), 27:(4,3), 28:(4,4), 29:(4,5), 30:(4,6), 31:(4,7),
                32:(3,0), 33:(3,1), 34:(3,2), 35:(3,3), 36:(3,4), 37:(3,5), 38:(3,6), 39:(3,7),
                40:(2,0), 41:(2,1), 42:(2,2), 43:(2,3), 44:(2,4), 45:(2,5), 46:(2,6), 47:(2,7),
                48:(1,0), 49:(1,1), 50:(1,2), 51:(1,3), 52:(1,4), 53:(1,5), 54:(1,6), 55:(1,7),
                56:(0,0), 57:(0,1), 58:(0,2), 59:(0,3), 60:(0,4), 61:(0,5), 62:(0,6), 63:(0,7)};

    def __init__(self,urlClone='https://github.com/iAmEthanMai/chess-engine-model.git',path=None):
        self.UrlClone=urlClone;
        if path is None:
            path=ChessModel.DefaultPath;
        else:
            path=str(path);

        if not exists(path):
            #descargo el modelo y lo guardo
            Repo.clone_from(urlClone, path);
        path_to_model = path + '/content/chess-engine-model/latest-model';    
        self.IAModel= tf.saved_model.load(path_to_model);

    def _find_best_moves(self,current_board,proportion = 0.5):
        """Return array of the best chess.Move
        
        Keyword arguments:
        current_board -- chess.Board()
        model -- tf.saved_model
        proportion -- proportion of best moves returned
        """
        moves = list(current_board.legal_moves)
        df_eval = ChessModel._get_possible_moves_data(current_board)
        predictions = self._predict(df_eval)
        good_move_probas = []
        
        for prediction in predictions:
            proto_tensor = tf.make_tensor_proto(prediction['probabilities'])
            proba = tf.make_ndarray(proto_tensor)[0][1]
            good_move_probas.append(proba)
            
        dict_ = dict(zip(moves, good_move_probas))
        dict_ = OrderedDict(sorted(dict_.items(), key = itemgetter(1), reverse = True))
        
        best_moves = list(dict_.keys())
    
        return best_moves[0:int(len(best_moves)*proportion)]

    def _predict(self,df_eval):
        """Return array of predictions for each row of df_eval
        
        Keyword arguments:
        df_eval -- pd.DataFrame
        """
        col_names = df_eval.columns
        dtypes = df_eval.dtypes
        predictions = []
        for row in df_eval.iterrows():
            example = tf.train.Example()
        for i in range(len(col_names)):
            dtype = dtypes[i]
            col_name = col_names[i]
            value = row[1][col_name]
            if dtype == 'object':
                value = bytes(value, 'utf-8')
                example.features.feature[col_name].bytes_list.value.extend([value])
            elif dtype == 'float':
                example.features.feature[col_name].float_list.value.extend([value])
            elif dtype == 'int':
                example.features.feature[col_name].int64_list.value.extend([value])
        predictions.append(self.IAModel.signatures['predict'](examples = tf.constant([example.SerializeToString()])))
        return predictions






    def _minimax(self,depth, board, alpha, beta, is_maximising_player,ai_white):
    
        if(depth == 0):
            return - ChessModel._evaluate_board(board,ai_white)
        elif(depth > 3):
            legal_moves = self._find_best_moves(board, 0.75)
        else:
            legal_moves = list(board.legal_moves)

        if(is_maximising_player):
            best_move = -9999
            for move in legal_moves:
                board.push(move)
                best_move = max(best_move, self._minimax(depth-1, board, alpha, beta, not is_maximising_player,ai_white))
                board.pop()
                alpha = max(alpha, best_move)
                if(beta <= alpha):
                    return best_move
            return best_move
        else:
            best_move = 9999
            for move in legal_moves:
                board.push(move)
                best_move = min(best_move, self._minimax(depth-1, board, alpha, beta, not is_maximising_player,ai_white))
                board.pop()
                beta = min(beta, best_move)
                if(beta <= alpha):
                    return best_move
            return best_move


    def _minimax_root(self,depth, board,ai_white, is_maximising_player = True):
    #only search the top 50% moves
        legal_moves = self._find_best_moves(board)
        best_move = -9999
        best_move_found = None

        for move in legal_moves:
            board.push(move)
            value = self._minimax(depth - 1, board, -10000, 10000, not is_maximising_player,ai_white)
            board.pop()
            if(value >= best_move):
                best_move = value
                best_move_found = move

        return best_move_found

    def Move(self,current_board,ai_white):
        for move in current_board.legal_moves:
            if(ChessModel._can_checkmate(move, current_board)):
                current_board.push(move)
                return

        nb_moves = len(list(current_board.legal_moves))
        
        if(nb_moves > 30):
            current_board.push(self._minimax_root(4, current_board,ai_white))
        elif(nb_moves > 10 and nb_moves <= 30):
            current_board.push(self._minimax_root(5, current_board,ai_white))
        else:
            current_board.push(self._minimax_root(7, current_board,ai_white))
        

    @staticmethod
    def _square_to_coord(square):

        return ChessModel._square_coord[square]

    @staticmethod
    def _get_piece_value(piece, square,ai_white):

        x, y = ChessModel._square_to_coord(square)
        
        if(ai_white):
            sign_white = -1
            sign_black = 1
        else:
            sign_white = 1
            sign_black = -1

        if(piece == 'None'):
            return 0
        elif(piece == 'P'):
            return sign_white * (10 + ChessModel._pawn_white_eval[x][y])
        elif(piece == 'N'):
            return sign_white * (30 + ChessModel._knight_white_eval[x][y])
        elif(piece == 'B'):
            return sign_white * (30 + ChessModel._bishop_white_eval[x][y])
        elif(piece == 'R'):
            return sign_white * (50 + ChessModel._rook_white_eval[x][y])
        elif(piece == 'Q'):
            return sign_white * (90 + ChessModel._queen_white_eval[x][y])
        elif(piece == 'K'):
            return sign_white * (900 + ChessModel._king_white_eval[x][y])
        elif(piece == 'p'):
            return sign_black * (10 + ChessModel._pawn_black_eval[x][y])
        elif(piece == 'n'):
            return sign_black * (30 + ChessModel._knight_black_eval[x][y])
        elif(piece == 'b'):
            return sign_black * (30 + ChessModel._bishop_black_eval[x][y])
        elif(piece == 'r'):
            return sign_black * (50 + ChessModel._rook_black_eval[x][y])
        elif(piece == 'q'):
            return sign_black * (90 + ChessModel._queen_black_eval[x][y])
        elif(piece == 'k'):
            return sign_black * (900 + ChessModel._king_black_eval[x][y])

    @staticmethod
    def _evaluate_board(board,ai_white):

        evaluation = 0
        for square in chess.SQUARES:
            piece = str(board.piece_at(square))
            evaluation = evaluation + ChessModel._get_piece_value(piece, square,ai_white)
        return evaluation

    @staticmethod
    def _can_checkmate(move, current_board):

        fen = current_board.fen()
        future_board = chess.Board(fen)
        future_board.push(move)
        return future_board.is_checkmate()
    @staticmethod    
    def _get_possible_moves_data(current_board):
        """Return pd.DataFrame of all possible moves used for predictions
        
        Keyword arguments:
        current_board -- chess.Board()
        """
        data = []
        moves = list(current_board.legal_moves)
        for move in moves:
            from_square, to_square = ChessModel._get_move_features(move)
            row = np.concatenate((ChessModel._get_board_features(current_board), from_square, to_square))
            data.append(row)
        
        board_feature_names = chess.SQUARE_NAMES
        move_from_feature_names = ['from_' + square for square in chess.SQUARE_NAMES]
        move_to_feature_names = ['to_' + square for square in chess.SQUARE_NAMES]
        
        columns = board_feature_names + move_from_feature_names + move_to_feature_names
        
        df = pd.DataFrame(data = data, columns = columns)

        for column in move_from_feature_names:
            df[column] = df[column].astype(float)
        for column in move_to_feature_names:
            df[column] = df[column].astype(float)
        return df

    @staticmethod
    def _get_board_features(board):
        """Return array of features for a board
        
        Keyword arguments:
        board -- chess.Board()
        """
        board_features = []
        for square in chess.SQUARES:
            board_features.append(str(board.piece_at(square)))
        return board_features

    @staticmethod
    def _get_move_features(move):
        """Return 2 arrays of features for a move
        
        Keyword arguments:
        move -- chess.Move
        """
        from_ = np.zeros(64)
        to_ = np.zeros(64)
        from_[move.from_square] = 1
        to_[move.to_square] = 1
        return from_, to_;
