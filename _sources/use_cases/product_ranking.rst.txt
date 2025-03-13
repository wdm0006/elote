Product Ranking with Elote
======================

Overview
--------

Product ranking is a common challenge in e-commerce, marketing, and user experience design. Traditional approaches often rely on explicit ratings (e.g., 1-5 stars) or engagement metrics (views, clicks), but these methods have limitations:

- Star ratings suffer from selection bias (mostly very satisfied or very dissatisfied users rate products)
- Engagement metrics can be misleading (high views might indicate misleading thumbnails rather than quality)
- Direct comparison data is often more reliable but harder to collect

Elote provides an elegant solution by implementing rating systems that can rank products based on head-to-head comparisons. This approach is particularly valuable for:

- A/B testing product variants
- Ranking items in a catalog based on user preferences
- Prioritizing features or improvements
- Building recommendation systems
- Optimizing product assortments

This guide demonstrates how to use Elote for product ranking through pairwise comparisons.

Basic Implementation
------------------

Let's start with a simple example where we want to rank different product variants based on user preferences:

.. code-block:: python

    from elote import LambdaArena
    import json
    
    # Sample data: tuples of (product_a, product_b, did_a_win)
    # In a real application, this would come from user choices
    comparisons = [
        ("Product A", "Product B", True),   # User preferred A over B
        ("Product A", "Product C", False),  # User preferred C over A
        ("Product B", "Product C", False),  # User preferred C over B
        ("Product A", "Product D", True),   # User preferred A over D
        ("Product B", "Product D", True),   # User preferred B over D
        ("Product C", "Product D", True),   # User preferred C over D
    ]
    
    # Define a function that returns True if the first product won
    def comparison_func(product_a, product_b, comparison_data=comparisons):
        for a, b, a_won in comparison_data:
            if a == product_a and b == product_b:
                return a_won
            elif a == product_b and b == product_a:
                return not a_won
        return None  # No data for this comparison
    
    # Create an arena with our comparison function
    arena = LambdaArena(comparison_func)
    
    # Process all the comparisons
    for product_a, product_b, _ in comparisons:
        arena.tournament([(product_a, product_b)])
    
    # Get the rankings
    rankings = arena.leaderboard()
    print(json.dumps(rankings, indent=2))

This would output a ranked list of products based on the pairwise comparisons.

Collecting Comparison Data
------------------------

There are several ways to collect pairwise comparison data:

1. **Direct User Feedback**: Present users with two options and ask which they prefer
2. **Implicit Behavior**: Track which product users choose when presented with alternatives
3. **Purchase Data**: Consider a purchase as a "win" against viewed-but-not-purchased alternatives
4. **Expert Evaluations**: Have product experts or focus groups compare items

Here's an example of a simple web interface for collecting direct user feedback:

.. code-block:: html

    <div class="product-comparison">
      <h2>Which product do you prefer?</h2>
      
      <div class="product-options">
        <div class="product" onclick="recordPreference('Product A')">
          <img src="product_a.jpg" alt="Product A">
          <h3>Product A</h3>
          <p>Description of Product A</p>
        </div>
        
        <div class="product" onclick="recordPreference('Product B')">
          <img src="product_b.jpg" alt="Product B">
          <h3>Product B</h3>
          <p>Description of Product B</p>
        </div>
      </div>
    </div>

    <script>
    function recordPreference(preferredProduct) {
      const otherProduct = preferredProduct === 'Product A' ? 'Product B' : 'Product A';
      
      // Send data to your backend
      fetch('/api/record-preference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          winner: preferredProduct,
          loser: otherProduct,
          timestamp: new Date().toISOString()
        })
      });
      
      // Load the next comparison
      loadNextComparison();
    }
    </script>

Advanced Implementation: E-commerce Product Ranking
------------------------------------------------

For a more realistic e-commerce scenario, we might want to:

1. Use a more sophisticated rating system like Glicko
2. Persist ratings between sessions
3. Handle new products with appropriate initial ratings

Here's a more complete example:

.. code-block:: python

    import pickle
    from elote import GlickoCompetitor
    
    class ProductRankingSystem:
        def __init__(self, ratings_file='product_ratings.pkl'):
            self.ratings_file = ratings_file
            self.products = self.load_ratings()
            
        def load_ratings(self):
            try:
                with open(self.ratings_file, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                return {}  # No existing ratings
                
        def save_ratings(self):
            with open(self.ratings_file, 'wb') as f:
                pickle.dump(self.products, f)
                
        def get_or_create_product(self, product_id, name=None, initial_rating=1500):
            if product_id not in self.products:
                self.products[product_id] = {
                    'competitor': GlickoCompetitor(initial_rating=initial_rating),
                    'name': name or product_id,
                    'id': product_id,
                    'comparisons': 0
                }
            return self.products[product_id]
            
        def record_comparison(self, winner_id, loser_id):
            winner = self.get_or_create_product(winner_id)
            loser = self.get_or_create_product(loser_id)
            
            # Update ratings
            winner['competitor'].beat(loser['competitor'])
            
            # Update comparison counts
            winner['comparisons'] += 1
            loser['comparisons'] += 1
            
            # Save updated ratings
            self.save_ratings()
            
        def get_rankings(self, min_comparisons=5):
            # Only rank products with sufficient data
            ranked_products = [p for p in self.products.values() if p['comparisons'] >= min_comparisons]
            
            # Sort by rating
            ranked_products.sort(key=lambda p: p['competitor'].rating, reverse=True)
            
            return [
                {
                    'id': p['id'],
                    'name': p['name'],
                    'rating': p['competitor'].rating,
                    'rd': p['competitor'].rd,  # Rating deviation (uncertainty)
                    'comparisons': p['comparisons']
                }
                for p in ranked_products
            ]
            
        def get_win_probability(self, product_a_id, product_b_id):
            product_a = self.get_or_create_product(product_a_id)
            product_b = self.get_or_create_product(product_b_id)
            
            return product_a['competitor'].expected_score(product_b['competitor'])
    
    # Usage example
    ranking_system = ProductRankingSystem()
    
    # Record some comparisons
    ranking_system.record_comparison('product-123', 'product-456')
    ranking_system.record_comparison('product-789', 'product-123')
    
    # Get current rankings
    rankings = ranking_system.get_rankings()
    for rank, product in enumerate(rankings, 1):
        print(f"{rank}. {product['name']} (Rating: {product['rating']:.1f} ± {product['rd']:.1f})")

Handling Cold Start
-----------------

New products have no comparison data, creating a "cold start" problem. Here are strategies to address this:

1. **Conservative Initial Ratings**: Start new products with average ratings but high uncertainty
2. **Exploration Phase**: Deliberately include new products in comparisons more frequently
3. **Metadata-Based Initialization**: Set initial ratings based on product attributes or similarity to existing products
4. **Expert Seeding**: Have product experts provide initial comparisons before user data is available

For example, to implement exploration-based comparisons:

.. code-block:: python

    def select_products_for_comparison(ranking_system, exploration_rate=0.3):
        """Select two products for comparison, balancing exploration and exploitation"""
        products = list(ranking_system.products.values())
        
        # Separate products with few comparisons
        new_products = [p for p in products if p['comparisons'] < 10]
        established_products = [p for p in products if p['comparisons'] >= 10]
        
        if not established_products:
            # Not enough data yet, just pick randomly
            import random
            return random.sample(products, 2) if len(products) >= 2 else None
            
        # Decide whether to explore or exploit
        import random
        if random.random() < exploration_rate and new_products:
            # Exploration: pair a new product with an established one
            product_a = random.choice(new_products)
            product_b = random.choice(established_products)
        else:
            # Exploitation: pick products with similar ratings for informative comparison
            product_a = random.choice(established_products)
            
            # Find products with similar ratings
            similar_products = sorted(
                [p for p in established_products if p['id'] != product_a['id']],
                key=lambda p: abs(p['competitor'].rating - product_a['competitor'].rating)
            )
            
            # Pick from the most similar ones
            product_b = random.choice(similar_products[:max(3, len(similar_products))])
            
        return product_a, product_b

Real-World Applications
---------------------

Companies using pairwise comparison approaches for product ranking include:

- **Netflix**: Uses pairwise comparisons to optimize thumbnail images for movies and shows
- **Booking.com**: Ranks hotel listings based on implicit pairwise preferences from user behavior
- **Spotify**: Uses similar techniques for playlist recommendations
- **Amazon**: Employs pairwise comparisons for product search ranking optimization

Conclusion
---------

Elote provides a powerful framework for implementing product ranking systems based on pairwise comparisons. This approach offers several advantages over traditional rating methods:

- More reliable data from direct comparisons
- Better handling of subjective preferences
- Reduced bias compared to explicit ratings
- Natural handling of new products through rating systems with uncertainty

By leveraging Elote's implementation of sophisticated rating systems, you can build robust product ranking solutions that more accurately reflect user preferences and improve the overall user experience. 