DROP TABLE IF EXISTS auctions CASCADE;
DROP TABLE IF EXISTS models CASCADE;
DROP TABLE IF EXISTS makes CASCADE;
DROP TABLE IF EXISTS transmissions CASCADE;
DROP TABLE IF EXISTS colors CASCADE;

CREATE TABLE IF NOT EXISTS makes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    make_id INTEGER REFERENCES makes(id),
    name VARCHAR(100) NOT NULL,
    UNIQUE(make_id, name)
);

CREATE TABLE IF NOT EXISTS transmissions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS colors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS auctions (
    id SERIAL PRIMARY KEY,
    year INTEGER,
    model_id INTEGER REFERENCES models(id),
    transmission_id INTEGER REFERENCES transmissions(id),
    exterior_color_id INTEGER REFERENCES colors(id),
    interior_color_id INTEGER REFERENCES colors(id),
    mileage INTEGER,
    title_status VARCHAR(100),
    location VARCHAR(200),
    state VARCHAR(50),
    engine VARCHAR(200),
    drivetrain VARCHAR(100),
    body_style VARCHAR(100),
    num_modifications INTEGER,
    sale_price NUMERIC,
    auction_date DATE,
    has_forced_induction INTEGER,
    url VARCHAR(255) UNIQUE
);
