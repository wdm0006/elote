Matchmaking with Elote
==================

Overview
--------

Matchmaking is the process of pairing players or competitors in a way that creates fair, balanced, and enjoyable matches. It's a critical component in competitive games, sports, and other contest formats. Poor matchmaking can lead to frustration, as matches that are too easy or too difficult are rarely satisfying for either participant.

Elote provides powerful tools for implementing sophisticated matchmaking systems through its rating implementations. This guide demonstrates how to use Elote for effective matchmaking in various scenarios.

Basic Matchmaking Principles
--------------------------

Effective matchmaking typically follows these principles:

1. **Skill-Based Pairing**: Match players with similar skill levels
2. **Uncertainty Handling**: Account for confidence in skill estimates
3. **Wait Time Balancing**: Balance match quality with queue time
4. **Variety**: Avoid repetitive matchups
5. **Team Balancing**: For team games, ensure teams have similar overall skill

Simple Skill-Based Matchmaking
----------------------------

Let's start with a basic implementation using Elo ratings:

.. code-block:: python

    from elote import EloCompetitor
    import random
    import math
    
    class MatchmakingSystem:
        def __init__(self):
            self.players = {}
            
        def register_player(self, player_id, initial_rating=1500):
            """Register a new player in the matchmaking system"""
            if player_id not in self.players:
                self.players[player_id] = {
                    'competitor': EloCompetitor(initial_rating=initial_rating),
                    'id': player_id,
                    'matches_played': 0
                }
            return self.players[player_id]
            
        def find_match(self, player_id, max_rating_diff=200):
            """Find a suitable opponent for a player"""
            if player_id not in self.players:
                self.register_player(player_id)
                
            player = self.players[player_id]
            player_rating = player['competitor'].rating
            
            # Filter players within rating range
            candidates = [
                p for pid, p in self.players.items()
                if pid != player_id and 
                abs(p['competitor'].rating - player_rating) <= max_rating_diff
            ]
            
            if not candidates:
                return None  # No suitable match found
                
            # Sort by rating difference (ascending)
            candidates.sort(key=lambda p: abs(p['competitor'].rating - player_rating))
            
            # Return the best match
            return candidates[0]['id']
            
        def record_match_result(self, winner_id, loser_id):
            """Record the result of a match"""
            if winner_id not in self.players:
                self.register_player(winner_id)
            if loser_id not in self.players:
                self.register_player(loser_id)
                
            # Update ratings
            self.players[winner_id]['competitor'].beat(self.players[loser_id]['competitor'])
            
            # Update match counts
            self.players[winner_id]['matches_played'] += 1
            self.players[loser_id]['matches_played'] += 1
            
        def get_player_stats(self, player_id):
            """Get stats for a player"""
            if player_id not in self.players:
                return None
                
            player = self.players[player_id]
            return {
                'id': player_id,
                'rating': player['competitor'].rating,
                'matches_played': player['matches_played']
            }
    
    # Usage example
    matchmaker = MatchmakingSystem()
    
    # Register some players
    matchmaker.register_player("player1", 1600)
    matchmaker.register_player("player2", 1550)
    matchmaker.register_player("player3", 1700)
    matchmaker.register_player("player4", 1400)
    
    # Find a match for player1
    opponent_id = matchmaker.find_match("player1")
    print(f"Found match: player1 vs {opponent_id}")
    
    # Record match result
    matchmaker.record_match_result("player1", opponent_id)
    
    # Check updated ratings
    print(f"Player1 rating: {matchmaker.get_player_stats('player1')['rating']}")
    print(f"Opponent rating: {matchmaker.get_player_stats(opponent_id)['rating']}")

Advanced Matchmaking with Glicko
------------------------------

Glicko is particularly well-suited for matchmaking because it tracks rating reliability through the rating deviation (RD) parameter. This allows us to account for uncertainty in player skill estimates:

.. code-block:: python

    from elote import GlickoCompetitor
    import math
    import time
    
    class GlickoMatchmaker:
        def __init__(self):
            self.players = {}
            
        def register_player(self, player_id, initial_rating=1500, initial_rd=350):
            """Register a new player with Glicko rating"""
            if player_id not in self.players:
                self.players[player_id] = {
                    'competitor': GlickoCompetitor(initial_rating=initial_rating, initial_rd=initial_rd),
                    'id': player_id,
                    'matches_played': 0,
                    'last_active': time.time()
                }
            return self.players[player_id]
            
        def find_match(self, player_id, max_rating_diff=300, consider_rd=True):
            """Find a match considering both rating and rating deviation"""
            if player_id not in self.players:
                self.register_player(player_id)
                
            player = self.players[player_id]
            player_rating = player['competitor'].rating
            player_rd = player['competitor'].rd
            
            # Update all players' RD based on inactivity
            self._update_inactive_players()
            
            candidates = []
            for pid, p in self.players.items():
                if pid == player_id:
                    continue
                    
                # Basic rating difference check
                rating_diff = abs(p['competitor'].rating - player_rating)
                if rating_diff > max_rating_diff:
                    continue
                
                # Calculate match quality score (lower is better)
                if consider_rd:
                    # Consider both rating difference and uncertainty
                    # Higher RD means we're less certain about the rating
                    combined_rd = math.sqrt(player_rd**2 + p['competitor'].rd**2)
                    match_quality = rating_diff / (1 + combined_rd/100)
                else:
                    match_quality = rating_diff
                    
                candidates.append((pid, match_quality))
            
            if not candidates:
                return None
                
            # Sort by match quality (ascending)
            candidates.sort(key=lambda x: x[1])
            
            # Return the best match
            return candidates[0][0]
            
        def _update_inactive_players(self):
            """Update RD for inactive players"""
            current_time = time.time()
            for player_id, player in self.players.items():
                # Calculate days since last activity
                days_inactive = (current_time - player['last_active']) / (60 * 60 * 24)
                if days_inactive > 1:
                    # Increase RD based on inactivity (simplified)
                    # In a real system, you'd use the proper Glicko formula
                    player['competitor'].rd = min(
                        350,  # Max RD
                        player['competitor'].rd + (days_inactive * 5)  # Increase by 5 per day
                    )
            
        def record_match_result(self, winner_id, loser_id):
            """Record match result and update ratings"""
            if winner_id not in self.players:
                self.register_player(winner_id)
            if loser_id not in self.players:
                self.register_player(loser_id)
                
            # Update ratings
            self.players[winner_id]['competitor'].beat(self.players[loser_id]['competitor'])
            
            # Update match counts and activity time
            current_time = time.time()
            for player_id in [winner_id, loser_id]:
                self.players[player_id]['matches_played'] += 1
                self.players[player_id]['last_active'] = current_time
                
        def get_player_stats(self, player_id):
            """Get detailed stats for a player"""
            if player_id not in self.players:
                return None
                
            player = self.players[player_id]
            return {
                'id': player_id,
                'rating': player['competitor'].rating,
                'rating_deviation': player['competitor'].rd,
                'matches_played': player['matches_played'],
                'last_active': player['last_active']
            }

Team-Based Matchmaking
--------------------

For team games, we need to consider the overall team skill level. Here's an approach using Elote:

.. code-block:: python

    from elote import EloCompetitor
    import random
    import statistics
    
    class TeamMatchmaker:
        def __init__(self):
            self.players = {}
            self.teams = {}
            
        def register_player(self, player_id, initial_rating=1500):
            """Register an individual player"""
            if player_id not in self.players:
                self.players[player_id] = {
                    'competitor': EloCompetitor(initial_rating=initial_rating),
                    'id': player_id,
                    'matches_played': 0
                }
            return self.players[player_id]
            
        def register_team(self, team_id, player_ids):
            """Register a team with its players"""
            if team_id not in self.teams:
                # Ensure all players are registered
                for player_id in player_ids:
                    if player_id not in self.players:
                        self.register_player(player_id)
                        
                # Create team
                self.teams[team_id] = {
                    'id': team_id,
                    'player_ids': player_ids,
                    'matches_played': 0,
                    'competitor': EloCompetitor(
                        initial_rating=self._calculate_team_rating(player_ids)
                    )
                }
            return self.teams[team_id]
            
        def _calculate_team_rating(self, player_ids):
            """Calculate a team's rating based on player ratings"""
            if not player_ids:
                return 1500
                
            ratings = [self.players[pid]['competitor'].rating for pid in player_ids]
            return statistics.mean(ratings)
            
        def find_team_match(self, team_id, max_rating_diff=200):
            """Find a suitable opponent team"""
            if team_id not in self.teams:
                return None
                
            team = self.teams[team_id]
            team_rating = team['competitor'].rating
            
            # Filter teams within rating range
            candidates = [
                t for tid, t in self.teams.items()
                if tid != team_id and 
                abs(t['competitor'].rating - team_rating) <= max_rating_diff
            ]
            
            if not candidates:
                return None
                
            # Sort by rating difference
            candidates.sort(key=lambda t: abs(t['competitor'].rating - team_rating))
            
            # Return the best match
            return candidates[0]['id']
            
        def record_team_match_result(self, winner_team_id, loser_team_id):
            """Record the result of a team match"""
            if winner_team_id not in self.teams or loser_team_id not in self.teams:
                return False
                
            # Update team ratings
            self.teams[winner_team_id]['competitor'].beat(
                self.teams[loser_team_id]['competitor']
            )
            
            # Update match counts
            self.teams[winner_team_id]['matches_played'] += 1
            self.teams[loser_team_id]['matches_played'] += 1
            
            # Also update individual player ratings based on their contribution
            # This is a simplified approach - real systems would consider individual performance
            for player_id in self.teams[winner_team_id]['player_ids']:
                for opponent_id in self.teams[loser_team_id]['player_ids']:
                    # Reduce the K-factor effect for team games
                    k_factor_original = self.players[player_id]['competitor'].k_factor
                    self.players[player_id]['competitor'].k_factor = k_factor_original / len(self.teams[winner_team_id]['player_ids'])
                    
                    # Update rating
                    self.players[player_id]['competitor'].beat(self.players[opponent_id]['competitor'])
                    
                    # Restore K-factor
                    self.players[player_id]['competitor'].k_factor = k_factor_original
                    
            return True

Matchmaking with Exploration
--------------------------

To avoid always matching the same players and to better learn player skills, we can incorporate exploration:

.. code-block:: python

    def find_match_with_exploration(self, player_id, max_rating_diff=300, exploration_rate=0.2):
        """Find a match with some randomness to explore player skills"""
        if player_id not in self.players:
            self.register_player(player_id)
            
        player = self.players[player_id]
        player_rating = player['competitor'].rating
        
        # Get all players within rating range
        candidates = [
            pid for pid, p in self.players.items()
            if pid != player_id and 
            abs(p['competitor'].rating - player_rating) <= max_rating_diff
        ]
        
        if not candidates:
            return None
            
        # With probability exploration_rate, pick a random candidate
        if random.random() < exploration_rate:
            return random.choice(candidates)
            
        # Otherwise, pick the closest match by rating
        candidates.sort(key=lambda pid: abs(self.players[pid]['competitor'].rating - player_rating))
        return candidates[0]

Real-World Applications
---------------------

Matchmaking systems using rating algorithms like those in Elote are used in:

- **Chess Platforms**: Chess.com, Lichess.org
- **Video Games**: League of Legends, Dota 2, Overwatch
- **Sports Leagues**: For scheduling matches and tournaments
- **Educational Platforms**: For pairing students of similar abilities
- **Competitive Programming**: Sites like Codeforces, TopCoder

Best Practices
------------

1. **Start Conservative**: Begin with wider rating ranges and narrow them as your player base grows
2. **Consider Wait Times**: Balance match quality with queue time
3. **Use Glicko for Sparse Data**: If players don't compete frequently, Glicko handles uncertainty better
4. **Incorporate Feedback**: Allow players to rate match quality
5. **Monitor Metrics**: Track win rates, match abandonment, and player satisfaction
6. **Adjust Dynamically**: Change parameters based on time of day, region, or queue length
7. **Handle New Players**: Use placement matches or conservative initial ratings

Conclusion
---------

Elote provides the foundation for building sophisticated matchmaking systems through its implementation of various rating algorithms. By leveraging these tools and following the principles outlined in this guide, you can create matchmaking systems that provide fair, balanced, and enjoyable experiences for your users. 