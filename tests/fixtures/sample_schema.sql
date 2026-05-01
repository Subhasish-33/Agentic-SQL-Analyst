CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE
);

CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    signup_source TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    category_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    price NUMERIC NOT NULL,
    inventory_count INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories (id)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    total NUMERIC NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price NUMERIC NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
);

INSERT INTO categories (id, name, slug) VALUES
    (1, 'Electronics', 'electronics'),
    (2, 'Home', 'home'),
    (3, 'Books', 'books'),
    (4, 'Fitness', 'fitness'),
    (5, 'Groceries', 'groceries');

INSERT INTO users (id, name, email, signup_source, created_at) VALUES
    (1, 'John Doe', 'john@example.com', 'organic', '2024-01-05 09:15:00'),
    (2, 'Priya Shah', 'priya@example.com', 'ads', '2024-01-12 11:20:00'),
    (3, 'Marcus Lee', 'marcus@example.com', 'referral', '2024-01-18 13:05:00'),
    (4, 'Sara Khan', 'sara@example.com', 'organic', '2024-02-02 10:45:00'),
    (5, 'Anita Rao', 'anita@example.com', 'ads', '2024-02-10 08:30:00'),
    (6, 'Lucas Chen', 'lucas@example.com', 'email', '2024-02-18 14:40:00'),
    (7, 'Emma Wilson', 'emma@example.com', 'organic', '2024-03-01 16:10:00'),
    (8, 'Raj Patel', 'raj@example.com', 'referral', '2024-03-11 12:25:00'),
    (9, 'Noah Kim', 'noah@example.com', 'ads', '2024-03-21 15:55:00'),
    (10, 'Mia Garcia', 'mia@example.com', 'organic', '2024-04-04 09:05:00'),
    (11, 'Olivia Brown', 'olivia@example.com', 'email', '2024-04-18 18:15:00'),
    (12, 'Ethan Davis', 'ethan@example.com', 'organic', '2024-05-02 07:50:00');

INSERT INTO products (id, category_id, name, price, inventory_count, created_at) VALUES
    (1, 1, 'Noise Cancelling Headphones', 199.99, 42, '2024-01-01 09:00:00'),
    (2, 1, 'Wireless Mouse', 39.50, 120, '2024-01-03 09:00:00'),
    (3, 2, 'Ceramic Lamp', 58.00, 34, '2024-01-05 09:00:00'),
    (4, 2, 'Standing Desk Mat', 72.25, 25, '2024-01-07 09:00:00'),
    (5, 3, 'Practical SQL', 32.00, 80, '2024-01-09 09:00:00'),
    (6, 3, 'Designing Data Systems', 44.00, 64, '2024-01-11 09:00:00'),
    (7, 4, 'Resistance Bands Set', 24.99, 90, '2024-01-13 09:00:00'),
    (8, 4, 'Yoga Mat', 29.99, 73, '2024-01-15 09:00:00'),
    (9, 5, 'Single Origin Coffee Beans', 18.50, 150, '2024-01-17 09:00:00'),
    (10, 5, 'Organic Oats', 9.75, 200, '2024-01-19 09:00:00'),
    (11, 1, '4K Monitor', 329.00, 18, '2024-01-21 09:00:00'),
    (12, 2, 'Air Purifier', 149.00, 27, '2024-01-23 09:00:00');

INSERT INTO orders (id, user_id, status, total, created_at) VALUES
    (1, 1, 'completed', 239.49, '2024-02-01 10:00:00'),
    (2, 2, 'completed', 58.00, '2024-02-03 12:10:00'),
    (3, 3, 'shipped', 76.00, '2024-02-05 14:25:00'),
    (4, 4, 'completed', 54.98, '2024-02-07 09:15:00'),
    (5, 5, 'processing', 29.99, '2024-02-09 18:45:00'),
    (6, 6, 'completed', 348.50, '2024-02-12 11:35:00'),
    (7, 7, 'completed', 62.49, '2024-02-15 16:20:00'),
    (8, 8, 'cancelled', 18.50, '2024-02-18 13:30:00'),
    (9, 9, 'completed', 149.00, '2024-02-22 08:40:00'),
    (10, 10, 'shipped', 82.25, '2024-02-25 17:05:00'),
    (11, 11, 'completed', 44.00, '2024-03-01 10:25:00'),
    (12, 12, 'processing', 209.74, '2024-03-03 15:55:00'),
    (13, 1, 'completed', 39.50, '2024-03-06 12:45:00'),
    (14, 2, 'completed', 338.75, '2024-03-09 09:50:00'),
    (15, 3, 'completed', 27.75, '2024-03-12 11:10:00'),
    (16, 4, 'processing', 181.00, '2024-03-15 14:05:00'),
    (17, 5, 'shipped', 72.25, '2024-03-18 16:35:00'),
    (18, 6, 'completed', 58.50, '2024-03-22 08:20:00'),
    (19, 7, 'completed', 32.00, '2024-03-25 19:40:00'),
    (20, 8, 'completed', 168.50, '2024-03-29 10:15:00');

INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    (1, 1, 1, 1, 199.99),
    (2, 1, 2, 1, 39.50),
    (3, 2, 3, 1, 58.00),
    (4, 3, 5, 1, 32.00),
    (5, 3, 6, 1, 44.00),
    (6, 4, 7, 1, 24.99),
    (7, 4, 8, 1, 29.99),
    (8, 5, 8, 1, 29.99),
    (9, 6, 11, 1, 329.00),
    (10, 6, 10, 2, 9.75),
    (11, 7, 7, 1, 24.99),
    (12, 7, 9, 2, 18.75),
    (13, 8, 9, 1, 18.50),
    (14, 9, 12, 1, 149.00),
    (15, 10, 4, 1, 72.25),
    (16, 10, 10, 1, 10.00),
    (17, 11, 6, 1, 44.00),
    (18, 12, 1, 1, 199.99),
    (19, 12, 10, 1, 9.75),
    (20, 13, 2, 1, 39.50),
    (21, 14, 11, 1, 329.00),
    (22, 14, 5, 1, 9.75),
    (23, 15, 10, 1, 9.75),
    (24, 15, 9, 1, 18.00),
    (25, 16, 12, 1, 149.00),
    (26, 16, 9, 1, 18.50),
    (27, 16, 10, 1, 13.50),
    (28, 17, 4, 1, 72.25),
    (29, 18, 2, 1, 39.50),
    (30, 18, 9, 1, 19.00),
    (31, 19, 5, 1, 32.00),
    (32, 20, 12, 1, 149.00),
    (33, 20, 9, 1, 19.50),
    (34, 1, 9, 1, 19.00),
    (35, 6, 2, 1, 19.00),
    (36, 14, 10, 1, 0.00);
