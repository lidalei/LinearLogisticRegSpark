name := "MPML"

version := "1.0"

scalaVersion := "2.11.8"

// add spark dependencies
val sparkVersion = "2.0.1"
libraryDependencies ++= Seq(
  //  groupID % artifactID % revision,
  //  %% will add scala version to artifactID
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.log4s" %% "log4s" % "1.3.3",
  "junit" % "junit" % "4.12"
)

mainClass in assembly := Some("regression.MyLinearRegression")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
    