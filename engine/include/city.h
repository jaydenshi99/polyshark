#pragma once

class City {
public:
    int  owner()      const { return _owner; }
    int  level()      const { return _level; }
    int  population() const { return _population; }
    int  tile_index() const { return _tile_index; }
    bool is_capital() const { return _is_capital; }

    void add_population(int n);
    void capture(int new_owner);

    void set_tile(int tile_index)  { _tile_index = tile_index; }
    void set_capital(bool capital) { _is_capital = capital; }

private:
    int  _owner      = -1;
    int  _level      = 1;
    int  _population = 0;
    int  _tile_index = -1;
    bool _is_capital = false;
};
