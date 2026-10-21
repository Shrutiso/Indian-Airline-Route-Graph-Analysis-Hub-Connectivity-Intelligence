ThisBuild / scalaVersion := "2.12.18"
ThisBuild / version := "1.0"

lazy val root = (project in file("."))
  .settings(
    name := "IndianAirlineGraphX",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "3.5.7",
      "org.apache.spark" %% "spark-sql" % "3.5.7",
      "org.apache.spark" %% "spark-graphx" % "3.5.7"
    )
  )
