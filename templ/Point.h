//
// Created by DEV on 2021/10/28.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

template<class T1, class T2>
class Point {
public:
    Point(T1 x, T2 y) : m_x_(x), m_y_(y){}

public:
    T1 GetX() const;
    void SetX(T1 x);
    T2 GetY() const;
    void SetY(T2 y);

private:
    T1 m_x_;
    T2 m_y_;
};

template<class T1, class T2>
T1 Point<T1, T2>::GetX() const {
    return m_x_;
}

template<class T1, class T2>
void Point<T1, T2>::SetX(T1 x){
    m_x_ = x;
}

template<class T1, class T2>
T2 Point<T1, T2>::GetY() const {
    return m_y_;
}

template<class T1, class T2>
void Point<T1, T2>::SetY(T2 y){
    m_y_ = y;
}

#endif //UNTITLED_POINT_H
