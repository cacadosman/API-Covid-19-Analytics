## API-Covid-19-Analytics
API untuk mengetahui informasi statistik dan prediksi dari kasus Covid-19.

[![GitHub version](https://img.shields.io/badge/version-0.2.1-yellow.svg)](https://github.com/cacadosman/API-Covid-19-Analytics/)
[![GitHub issues](https://img.shields.io/github/issues/cacadosman/api-covid-19-analytics)](https://github.com/cacadosman/API-Covid-19-Analytics/issues)
[![GitHub license](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/cacadosman/API-Covid-19-Analytics/blob/master/LICENSE)

### List API

#### Get prediction of cumulative cases for next 3 days in Indonesia
- Endpoint : `/api/v1/predict` <br>
- HTTP Method : `GET` <br>
- Filter : `-`
- Pagination : `-`
- Request Header :
    - Accept : `application/json`
    - Content-type : `application/json`
- Request Body : `-`
- Response Body : <br>
```json
{
    "data": {
        "from": 25,
        "to": 27,
        "values": [
            899,
            1016,
            1142
        ]
    },
    "success": true
}
```

***API ini masih dalam tahap pengembangan dan akan terus dikembangkan.**

Kontak:
- FB: cacadosman23
- Discord: cacadosman#3356
- Email: fadli.m@mail.ugm.ac.id


~ Made by love.
Terima kasih.
