import math

k_factor = 32

def excepted_score(rating1, rating2):
    return 1/ (1 + math.pow(10, (rating2 - rating1)/ 400))

def update_rating(rating, excepted_score, actual_score, k):
    return rating + k * (actual_score - excepted_score)


def elo_cal(player1_rating, player2_rating, player1_socre, player2_socre, k_factor):
    excepted1_score = excepted_score(player1_rating, player2_rating)
    excepted2_score = excepted_score(player2_rating, player1_rating)

    new_player1_rating = update_rating(player1_rating, excepted1_score, player1_socre, k_factor)
    new_player2_rating = update_rating(player2_rating, excepted2_score, player2_socre, k_factor)
    return new_player1_rating, new_player2_rating

if __name__=='__main__':
    player1_rating = 1300
    player2_rating = 1300
    player1_socre = 1
    player2_socre = 0

    for i in range(10):
        player1_rating, player2_rating = elo_cal(player1_rating, player2_rating, player1_socre, player2_socre, k_factor)

    print(player1_rating, player2_rating)
