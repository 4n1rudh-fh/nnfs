"""Example of a class and a object instantiation."""

class Pizza:
    """Models the idea of a pizza."""

    def __init__(self, size: str, n_topping: int, extra_cheese: bool):
        self.size = size
        self.n_topping = n_topping
        self.extra_cheese = extra_cheese

    def price(self, tax: float) -> float:
        """Calculates price of a pizza"""
        total: float = 0.0
        if self.size == "large":
            total += 10.0
        else:
            total += 8.0
        
        if self.extra_cheese:
            total += 1.0
        
        total += self.n_topping * 0.75

        # Taxes
        total += total * tax
        
        return total

a_pizza: Pizza = Pizza("large", 3, False)
print(Pizza)
print(a_pizza)
print(a_pizza.size)
print(f"Prize: {a_pizza.price(1.05)} €")

b_pizza: Pizza = Pizza("small", 0, True)
print(f"Price: {b_pizza.price(1.05)} €")
